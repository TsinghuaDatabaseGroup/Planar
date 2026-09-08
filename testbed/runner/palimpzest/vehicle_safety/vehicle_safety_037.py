#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-037."""

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

TASK_ID = "vehicle_safety-037"
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
    "root_cause_theme",
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

        acura_complaints = complaints.loc[
            (complaints["make"] == "ACURA")
            & (complaints["component_id"] == "ENGINE"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), acura_complaints)

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

        acura_recalls = recalls.loc[
            (recalls["vehicle_make"] == "ACURA")
            & (recalls["recall_component_id"] == "ENGINE"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), acura_recalls)

        pair_plan = memory_dataset(
            f"{TASK_ID}-complaints", acura_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", acura_recalls),
            condition=(
                "Within Acura ENGINE records, match a complaint to a recall only "
                "when the complaint narrative and recall defect summary describe "
                "the same engine failure mechanism."
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
            {"left": len(acura_complaints), "right": len(acura_recalls)},
            complaint_recall_pairs,
            pair_result,
            time.time() - started,
        )

        root_plan = memory_dataset(
            f"{TASK_ID}-root-cause", complaint_recall_pairs
        ).sem_map(
            cols=[
                {
                    "name": "root_cause_theme",
                    "type": str,
                    "desc": "The shared engine root-cause theme in at most eight words.",
                }
            ],
            desc="Extract the root-cause theme shared by the complaint and recall.",
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        root_result = root_plan.run(config)
        extracted_pairs = result_frame(
            root_result,
            complaint_recall_pairs,
            ["root_cause_theme"],
        )
        tracker.record_semantic(
            "sem_map",
            len(complaint_recall_pairs),
            extracted_pairs,
            root_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety", extracted_pairs
        ).sem_filter(
            filter=(
                "The matched complaint and recall consequence describe loss of "
                "motive power, stalling, engine seizure, or another driving-safety "
                "consequence."
            ),
            depends_on=["summary_text", "recall_consequence_summary"],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        safety_pairs = result_frame(safety_result, extracted_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(extracted_pairs),
            safety_pairs,
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
                r"Acura|Honda|engine failure",
                case=False,
                na=False,
                regex=True,
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(reports), candidate_reports)

        report_left = safety_pairs[
            [
                "complaint_id",
                "campaign_number",
                "summary_text",
                "recall_defect_summary",
                "root_cause_theme",
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
                "Match the complaint-recall root-cause theme to an investigation "
                "report only when the report discusses the same Acura or Honda "
                "engine-failure scope or ODI finding."
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
                    "root_cause_theme",
                    "matched_complaint_count",
                    "distinct_campaign_count",
                    "related_report_ids",
                    "report_bodies",
                ]
            )
        else:
            grouped = (
                report_matches.groupby(
                    "root_cause_theme",
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

        evidence_rows = []
        for index, row in enumerate(
            grouped.to_dict(orient="records"),
            start=1,
        ):
            aggregate_input = pd.DataFrame(
                [
                    {
                        "root_cause_theme": row["root_cause_theme"],
                        "report_bodies": row["report_bodies"],
                    }
                ]
            )
            aggregate_plan = memory_dataset(
                f"{TASK_ID}-aggregate-{index}", aggregate_input
            ).sem_agg(
                col={
                    "name": "odi_evidence_summary",
                    "type": str,
                    "desc": "A single short phrase summarizing the ODI evidence.",
                },
                agg=(
                    "Summarize the ODI evidence across the related investigation "
                    "report bodies in one short phrase."
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
                    f"{TASK_ID}: semantic aggregate returned no evidence summary"
                )
            evidence_rows.append(
                {
                    "root_cause_theme": row["root_cause_theme"],
                    "odi_evidence_summary": aggregate_frame.iloc[0][
                        "odi_evidence_summary"
                    ],
                }
            )

        evidence = pd.DataFrame.from_records(
            evidence_rows,
            columns=["root_cause_theme", "odi_evidence_summary"],
        )
        with_evidence = grouped.merge(
            evidence,
            on="root_cause_theme",
            how="inner",
        )
        ordered = with_evidence.sort_values(
            ["matched_complaint_count", "root_cause_theme"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(with_evidence), ordered)

        leading = ordered.head(1).reset_index(drop=True)
        tracker.record("limit", len(ordered), leading)

        result = leading[
            [
                "root_cause_theme",
                "matched_complaint_count",
                "distinct_campaign_count",
                "related_report_ids",
                "odi_evidence_summary",
            ]
        ].copy()
        tracker.record("project", len(leading), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
