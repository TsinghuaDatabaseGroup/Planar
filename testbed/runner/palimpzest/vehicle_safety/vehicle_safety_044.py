#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-044."""

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
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "vehicle_safety-044"
DATASET = "nhtsa_vehicle_safety"
RECALL_SEVERITIES = ("critical", "severe", "moderate", "minor")
MATCH_COLUMNS = [
    "complaint_id",
    "component_id",
    "crash_flag",
    "injury_count",
    "vehicle_towed_flag",
    "summary_text",
    "campaign_number",
    "recall_component_id",
    "recall_defect_summary",
    "recall_consequence_summary",
]


def normalize_failure_mode(value) -> str:
    return " ".join(str(value).strip().removesuffix(".").lower().split())


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            [
                "complaint_id",
                "make",
                "component_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "summary_text",
            ],
        )
        tracker.record("scan", None, complaints)

        acura_complaints = complaints.loc[
            (complaints["make"] == "ACURA")
            & (complaints["component_id"] == "BRAKES"),
            [
                "complaint_id",
                "component_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "summary_text",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), acura_complaints)

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

        acura_recalls = recalls.loc[
            (recalls["vehicle_make"] == "ACURA")
            & (recalls["recall_component_id"] == "BRAKES"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), acura_recalls)

        match_plan = memory_dataset(
            f"{TASK_ID}-complaints", acura_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", acura_recalls),
            condition=(
                "Within Acura BRAKES records, match a complaint to a recall only "
                "when the complaint narrative and recall defect summary describe "
                "the same braking defect."
            ),
            depends_on=MATCH_COLUMNS,
        )
        started = time.time()
        match_result = match_plan.run(config)
        matched_pairs = result_frame(match_result)
        if matched_pairs.empty:
            matched_pairs = pd.DataFrame(columns=MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(acura_complaints), "right": len(acura_recalls)},
            matched_pairs,
            match_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety", matched_pairs
        ).sem_filter(
            filter=(
                "The complaint describes a real braking-performance safety issue "
                "such as loss of braking, extended stopping distance, unintended "
                "braking, or brake failure."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        braking_issues = result_frame(safety_result, matched_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(matched_pairs),
            braking_issues,
            safety_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-failure-mode", braking_issues
        ).sem_map(
            cols=[
                {
                    "name": "braking_failure_mode",
                    "type": str,
                    "desc": (
                        "The dominant braking failure mode shared by the complaint "
                        "and recall, in at most six words."
                    ),
                }
            ],
            desc="Extract the dominant braking failure mode.",
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            braking_issues,
            ["braking_failure_mode"],
        )
        extracted["braking_failure_mode"] = extracted[
            "braking_failure_mode"
        ].map(normalize_failure_mode)
        tracker.record_semantic(
            "sem_map",
            len(braking_issues),
            extracted,
            extraction_result,
            time.time() - started,
        )

        severity_plan = memory_dataset(
            f"{TASK_ID}-severity", extracted
        ).sem_map(
            cols=[
                {
                    "name": "recall_severity",
                    "type": str,
                    "desc": (
                        "Exactly one of critical, severe, moderate, or minor."
                    ),
                }
            ],
            desc="Assign the recall severity.",
            depends_on=["recall_consequence_summary"],
        )
        started = time.time()
        severity_result = severity_plan.run(config)
        severity_labeled = result_frame(
            severity_result,
            extracted,
            ["recall_severity"],
        )
        severity_labeled["recall_severity"] = severity_labeled[
            "recall_severity"
        ].map(lambda value: normalize_enum(value, RECALL_SEVERITIES))
        if severity_labeled["recall_severity"].isna().any():
            raise ValueError(f"{TASK_ID}: invalid recall severity label")
        tracker.record_semantic(
            "sem_map",
            len(extracted),
            severity_labeled,
            severity_result,
            time.time() - started,
        )

        aggregation_input = severity_labeled.copy()
        aggregation_input["severe_complaint"] = (
            aggregation_input["crash_flag"]
            | aggregation_input["injury_count"].gt(0)
            | aggregation_input["vehicle_towed_flag"]
        )
        if aggregation_input.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "severe_complaint_rate",
                    "dominant_braking_failure_mode",
                    "dominant_recall_severity",
                ]
            )
        else:
            grouped = (
                aggregation_input.groupby(
                    "campaign_number",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    matched_complaint_count=("complaint_id", "nunique"),
                    severe_complaint_rate=("severe_complaint", "mean"),
                    dominant_braking_failure_mode=(
                        "braking_failure_mode",
                        stable_mode,
                    ),
                    dominant_recall_severity=(
                        "recall_severity",
                        stable_mode,
                    ),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(aggregation_input), grouped)

        ordered = grouped.sort_values(
            ["matched_complaint_count", "campaign_number"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)

        top_three = ordered.head(3).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_three)

        top_three.insert(0, "rank", range(1, len(top_three) + 1))
        result = top_three[[
            "rank",
            "campaign_number",
            "matched_complaint_count",
            "severe_complaint_rate",
            "dominant_braking_failure_mode",
            "dominant_recall_severity",
        ]].copy()
        tracker.record("project", len(top_three), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
