#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-042."""

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
    normalize_enum,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "vehicle_safety-042"
DATASET = "nhtsa_vehicle_safety"
ADAS_TOPICS = (
    "autopilot_misuse",
    "unexpected_braking",
    "steering_control",
    "warning_or_supervision_gap",
    "other",
)
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
    "adas_topic",
    "file_id",
    "body",
]


def distinct_sorted(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values.dropna()})


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

        tesla_complaints = complaints.loc[
            (complaints["make"] == "TESLA")
            & (complaints["component_id"] == "ADAS"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), tesla_complaints)

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

        tesla_recalls = recalls.loc[
            (recalls["vehicle_make"] == "TESLA")
            & (recalls["recall_component_id"] == "ADAS"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), tesla_recalls)

        pair_plan = memory_dataset(
            f"{TASK_ID}-complaints", tesla_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", tesla_recalls),
            condition=(
                "Within Tesla ADAS records, match a complaint to a recall only "
                "when the complaint narrative and recall defect summary describe "
                "the same driver-assistance or automation-control problem."
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
            {"left": len(tesla_complaints), "right": len(tesla_recalls)},
            complaint_recall_pairs,
            pair_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety", complaint_recall_pairs
        ).sem_filter(
            filter=(
                "The complaint describes a driving-context ADAS safety issue "
                "rather than general infotainment or account software behavior."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        driving_issues = result_frame(safety_result, complaint_recall_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(complaint_recall_pairs),
            driving_issues,
            safety_result,
            time.time() - started,
        )

        topic_plan = memory_dataset(
            f"{TASK_ID}-topic", driving_issues
        ).sem_map(
            cols=[
                {
                    "name": "adas_topic",
                    "type": str,
                    "desc": (
                        "Exactly one of autopilot_misuse, unexpected_braking, "
                        "steering_control, warning_or_supervision_gap, or other."
                    ),
                }
            ],
            desc="Assign the dominant Tesla ADAS defect topic.",
            depends_on=["summary_text"],
        )
        started = time.time()
        topic_result = topic_plan.run(config)
        topic_labeled = result_frame(
            topic_result,
            driving_issues,
            ["adas_topic"],
        )
        topic_labeled["adas_topic"] = topic_labeled["adas_topic"].map(
            lambda value: normalize_enum(value, ADAS_TOPICS)
        )
        if topic_labeled["adas_topic"].isna().any():
            raise ValueError(f"{TASK_ID}: invalid ADAS topic label")
        tracker.record_semantic(
            "sem_map",
            len(driving_issues),
            topic_labeled,
            topic_result,
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
                r"Tesla|Autopilot|driver assistance",
                case=False,
                na=False,
                regex=True,
            ),
            ["file_id", "body"],
        ].reset_index(drop=True)
        tracker.record("filter", len(reports), candidate_reports)

        report_left = topic_labeled[[
            "complaint_id",
            "campaign_number",
            "summary_text",
            "recall_defect_summary",
            "adas_topic",
        ]].reset_index(drop=True)
        report_plan = memory_dataset(
            f"{TASK_ID}-report-left", report_left
        ).sem_join(
            memory_dataset(f"{TASK_ID}-report-right", candidate_reports),
            condition=(
                "Match the Tesla ADAS complaint-recall issue to an investigation "
                "report only when the report discusses the same driver-assistance "
                "safety issue or ODI human-factors concern."
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
                    "campaign_number",
                    "matched_complaint_count",
                    "dominant_adas_topic",
                    "related_report_ids",
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
                    dominant_adas_topic=("adas_topic", stable_mode),
                    related_report_ids=("file_id", distinct_sorted),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(report_matches), grouped)

        result = grouped[[
            "campaign_number",
            "matched_complaint_count",
            "dominant_adas_topic",
            "related_report_ids",
        ]].copy()
        tracker.record("project", len(grouped), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
