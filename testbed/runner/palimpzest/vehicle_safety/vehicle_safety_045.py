#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-045."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_jsonl,
    load_table,
    load_text_documents,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-045"
DATASET = "nhtsa_vehicle_safety"
COMPLAINT_RECALL_COLUMNS = [
    "complaint_id",
    "component_id",
    "summary_text",
    "campaign_number",
    "recall_component_id",
    "recall_defect_summary",
    "recall_consequence_summary",
]
REPORT_MATCH_COLUMNS = [
    "complaint_id",
    "campaign_number",
    "summary_text",
    "recall_defect_summary",
    "defect_theme",
    "file_id",
    "body",
]


def distinct_sorted(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values.dropna()})


def distinct_in_order(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values.dropna()))


def normalize_theme(value) -> str:
    return " ".join(str(value).strip().removesuffix(".").lower().split())


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        chevrolet_complaints = complaints.loc[
            (complaints["make"] == "CHEVROLET")
            & (complaints["component_id"] == "BRAKES"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), chevrolet_complaints)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "vehicle_make",
            "component_component_id",
            "risk_defect_summary",
            "risk_consequence_summary",
        ]].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
                "risk_consequence_summary": "recall_consequence_summary",
            }
        )
        tracker.record("scan", None, recalls)

        chevrolet_recalls = recalls.loc[
            (recalls["vehicle_make"] == "CHEVROLET")
            & (recalls["recall_component_id"] == "BRAKES"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), chevrolet_recalls)

        pair_plan = memory_dataset(
            f"{TASK_ID}-complaints", chevrolet_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", chevrolet_recalls),
            condition=(
                "Within Chevrolet BRAKES records, match a complaint to a recall "
                "only when the complaint narrative and recall defect summary "
                "describe the same braking defect or risk."
            ),
            depends_on=COMPLAINT_RECALL_COLUMNS,
        )
        started = time.time()
        pair_result = pair_plan.run(config)
        complaint_recall_pairs = result_frame(pair_result)
        if complaint_recall_pairs.empty:
            complaint_recall_pairs = pd.DataFrame(
                columns=COMPLAINT_RECALL_COLUMNS
            )
        tracker.record_semantic(
            "sem_join",
            {
                "left": len(chevrolet_complaints),
                "right": len(chevrolet_recalls),
            },
            complaint_recall_pairs,
            pair_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-theme", complaint_recall_pairs
        ).sem_map(
            cols=[
                {
                    "name": "defect_theme",
                    "type": str,
                    "desc": (
                        "The braking defect theme shared by the complaint and "
                        "recall, in at most eight words."
                    ),
                }
            ],
            desc="Extract the shared braking defect theme.",
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            complaint_recall_pairs,
            ["defect_theme"],
        )
        extracted["defect_theme"] = extracted["defect_theme"].map(
            normalize_theme
        )
        tracker.record_semantic(
            "sem_map",
            len(complaint_recall_pairs),
            extracted,
            extraction_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety", extracted
        ).sem_filter(
            filter=(
                "The matched evidence describes a braking safety risk with a "
                "potential crash, loss-of-control, or extended-stopping "
                "consequence."
            ),
            depends_on=["summary_text", "recall_consequence_summary"],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        safety_matches = result_frame(safety_result, extracted)
        tracker.record_semantic(
            "sem_filter",
            len(extracted),
            safety_matches,
            safety_result,
            time.time() - started,
        )

        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        tracker.record("scan", None, reports)

        candidate_reports = reports.loc[
            reports["body"].str.contains(
                r"Chevrolet|GM|brake",
                case=False,
                na=False,
                regex=True,
            ),
            ["file_id", "body"],
        ].reset_index(drop=True)
        tracker.record("filter", len(reports), candidate_reports)

        report_left = safety_matches[[
            "complaint_id",
            "campaign_number",
            "summary_text",
            "recall_defect_summary",
            "defect_theme",
        ]].reset_index(drop=True)
        report_plan = memory_dataset(
            f"{TASK_ID}-report-left", report_left
        ).sem_join(
            memory_dataset(f"{TASK_ID}-report-right", candidate_reports),
            condition=(
                "Match the braking defect theme to an investigation report only "
                "when the report discusses the same Chevrolet or GM braking "
                "issue or ODI risk finding."
            ),
            depends_on=REPORT_MATCH_COLUMNS,
        )
        started = time.time()
        report_result = report_plan.run(config)
        report_matches = result_frame(report_result)
        if report_matches.empty:
            report_matches = pd.DataFrame(columns=REPORT_MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(report_left), "right": len(candidate_reports)},
            report_matches,
            report_result,
            time.time() - started,
        )

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "defect_theme",
                    "matched_complaint_count",
                    "distinct_campaign_count",
                    "related_report_ids",
                    "report_bodies",
                ]
            )
        else:
            grouped = (
                report_matches.groupby(
                    "defect_theme",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    matched_complaint_count=("complaint_id", "nunique"),
                    distinct_campaign_count=("campaign_number", "nunique"),
                    related_report_ids=("file_id", distinct_sorted),
                    report_bodies=("body", distinct_in_order),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(report_matches), grouped)

        summary_rows = []
        for index, row in enumerate(
            grouped.to_dict(orient="records"),
            start=1,
        ):
            aggregate_input = pd.DataFrame(
                [{"report_bodies": row["report_bodies"]}]
            )
            aggregate_plan = memory_dataset(
                f"{TASK_ID}-aggregate-{index}", aggregate_input
            ).sem_agg(
                col={
                    "name": "odi_risk_summary",
                    "type": str,
                    "desc": (
                        "A single short phrase summarizing the ODI braking risk "
                        "finding."
                    ),
                },
                agg=(
                    "Summarize the ODI braking risk finding across the related "
                    "investigation report bodies in one short phrase."
                ),
                depends_on=["report_bodies"],
            )
            started = time.time()
            aggregate_result = aggregate_plan.run(config)
            aggregate_frame = result_frame(aggregate_result)
            tracker.record_semantic(
                "sem_agg",
                len(aggregate_input),
                aggregate_frame,
                aggregate_result,
                time.time() - started,
            )
            if aggregate_frame.empty:
                raise ValueError(
                    f"{TASK_ID}: semantic aggregate returned no risk summary"
                )
            summary_rows.append(
                {
                    "defect_theme": row["defect_theme"],
                    "odi_risk_summary": aggregate_frame.iloc[0][
                        "odi_risk_summary"
                    ],
                }
            )

        summaries = pd.DataFrame.from_records(
            summary_rows,
            columns=["defect_theme", "odi_risk_summary"],
        )
        with_summaries = grouped.merge(
            summaries,
            on="defect_theme",
            how="inner",
        )
        ordered = with_summaries.sort_values(
            ["matched_complaint_count", "defect_theme"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(with_summaries), ordered)

        top_two = ordered.head(2).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_two)

        top_two.insert(0, "rank", range(1, len(top_two) + 1))
        result = top_two[[
            "rank",
            "defect_theme",
            "matched_complaint_count",
            "distinct_campaign_count",
            "related_report_ids",
            "odi_risk_summary",
        ]].copy()
        tracker.record("project", len(top_two), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
