#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-035."""

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
    stable_mode,
)

TASK_ID = "vehicle_safety-035"
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
    "root_cause_phrase",
    "file_id",
    "body",
]


def distinct_sorted(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values.dropna()})


def distinct_in_order(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values.dropna()))


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

        dodge_complaints = complaints.loc[
            (complaints["make"] == "DODGE")
            & (complaints["component_id"] == "AIRBAG"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), dodge_complaints)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
                "risk_consequence_summary",
            ]
        ].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
                "risk_consequence_summary": "recall_consequence_summary",
            }
        )
        tracker.record("scan", None, recalls)

        dodge_recalls = recalls.loc[
            (recalls["vehicle_make"] == "DODGE")
            & (recalls["recall_component_id"] == "AIRBAG"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), dodge_recalls)

        pair_plan = memory_dataset(
            f"{TASK_ID}-complaints", dodge_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", dodge_recalls),
            condition=(
                "Within Dodge AIRBAG records, match a complaint to a recall only "
                "when the complaint narrative and recall defect summary describe "
                "the same airbag failure mechanism or deployment hazard."
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
            {"left": len(dodge_complaints), "right": len(dodge_recalls)},
            complaint_recall_pairs,
            pair_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety-pairs", complaint_recall_pairs
        ).sem_filter(
            filter=(
                "The matched complaint-recall pair describes an airbag safety "
                "consequence such as non-deployment, unintended deployment, "
                "inflator rupture, injury, or crash exposure."
            ),
            depends_on=[
                "summary_text",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        safety_pairs = result_frame(safety_result, complaint_recall_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(complaint_recall_pairs),
            safety_pairs,
            safety_result,
            time.time() - started,
        )

        root_plan = memory_dataset(
            f"{TASK_ID}-root-cause", safety_pairs
        ).sem_map(
            cols=[
                {
                    "name": "root_cause_phrase",
                    "type": str,
                    "desc": (
                        "A short root-cause phrase shared by the complaint and "
                        "recall defect descriptions."
                    ),
                }
            ],
            desc="Extract the shared root cause from the matched descriptions.",
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        root_result = root_plan.run(config)
        extracted_pairs = result_frame(
            root_result,
            safety_pairs,
            ["root_cause_phrase"],
        )
        tracker.record_semantic(
            "sem_map",
            len(safety_pairs),
            extracted_pairs,
            root_result,
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
                r"Dodge|airbag|inflator",
                case=False,
                na=False,
                regex=True,
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(reports), candidate_reports)

        report_left = extracted_pairs[
            [
                "complaint_id",
                "campaign_number",
                "summary_text",
                "recall_defect_summary",
                "root_cause_phrase",
            ]
        ].reset_index(drop=True)
        report_right = candidate_reports[["file_id", "body"]].reset_index(
            drop=True
        )
        report_plan = memory_dataset(
            f"{TASK_ID}-report-left", report_left
        ).sem_join(
            memory_dataset(f"{TASK_ID}-report-right", report_right),
            condition=(
                "Match the Dodge airbag complaint-recall defect to an "
                "investigation report only when the report discusses the same "
                "airbag mechanism and ODI safety assessment."
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
            {"left": len(report_left), "right": len(report_right)},
            report_matches,
            report_result,
            time.time() - started,
        )

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "related_report_ids",
                    "root_cause_phrase",
                    "report_bodies",
                ]
            )
        else:
            grouped = (
                report_matches.groupby(
                    "campaign_number",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    matched_complaint_count=("complaint_id", "nunique"),
                    related_report_ids=("file_id", distinct_sorted),
                    root_cause_phrase=("root_cause_phrase", stable_mode),
                    report_bodies=("body", distinct_in_order),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(report_matches), grouped)

        conclusion_rows = []
        for index, row in enumerate(
            grouped.to_dict(orient="records"),
            start=1,
        ):
            aggregate_input = pd.DataFrame(
                [
                    {
                        "campaign_number": row["campaign_number"],
                        "report_bodies": row["report_bodies"],
                    }
                ]
            )
            aggregate_plan = memory_dataset(
                f"{TASK_ID}-aggregate-{index}", aggregate_input
            ).sem_agg(
                col={
                    "name": "odi_safety_conclusion",
                    "type": str,
                    "desc": "A single short phrase summarizing the ODI safety conclusion.",
                },
                agg=(
                    "Summarize the ODI safety conclusion for this campaign from "
                    "the related investigation report bodies in one short phrase."
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
                    f"{TASK_ID}: semantic aggregate returned no conclusion"
                )
            conclusion_rows.append(
                {
                    "campaign_number": row["campaign_number"],
                    "odi_safety_conclusion": aggregate_frame.iloc[0][
                        "odi_safety_conclusion"
                    ],
                }
            )

        conclusions = pd.DataFrame.from_records(
            conclusion_rows,
            columns=["campaign_number", "odi_safety_conclusion"],
        )
        result = grouped.merge(
            conclusions,
            on="campaign_number",
            how="inner",
        )[
            [
                "campaign_number",
                "matched_complaint_count",
                "related_report_ids",
                "root_cause_phrase",
                "odi_safety_conclusion",
            ]
        ].copy()
        tracker.record("project", len(grouped), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
