#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-036."""

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
    parse_bool,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "vehicle_safety-036"
DATASET = "nhtsa_vehicle_safety"
COMPLAINT_SEVERITIES = (
    "minor_inconvenience",
    "moderate_safety_event",
    "severe_safety_event",
)
REMEDY_STYLES = ("software_update", "part_replacement", "other")
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
    "recall_corrective_action",
]


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

        ford_complaints = complaints.loc[
            (complaints["make"] == "FORD")
            & (complaints["component_id"] == "AIRBAG"),
            [
                "complaint_id",
                "component_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "summary_text",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), ford_complaints)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
                "remedy_corrective_action",
            ]
        ].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
                "remedy_corrective_action": "recall_corrective_action",
            }
        )
        tracker.record("scan", None, recalls)

        ford_recalls = recalls.loc[
            (recalls["vehicle_make"] == "FORD")
            & (recalls["recall_component_id"] == "AIRBAG"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_corrective_action",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), ford_recalls)

        match_plan = memory_dataset(
            f"{TASK_ID}-complaints", ford_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", ford_recalls),
            condition=(
                "Within Ford AIRBAG records, match a complaint to a recall only "
                "when the complaint narrative and recall defect summary describe "
                "the same airbag failure mechanism."
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
            {"left": len(ford_complaints), "right": len(ford_recalls)},
            matched_pairs,
            match_result,
            time.time() - started,
        )

        event_plan = memory_dataset(
            f"{TASK_ID}-events", matched_pairs
        ).sem_filter(
            filter=(
                "The complaint narrative describes a concrete airbag safety event "
                "rather than a cosmetic, warning-light-only concern."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        event_result = event_plan.run(config)
        concrete_events = result_frame(event_result, matched_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(matched_pairs),
            concrete_events,
            event_result,
            time.time() - started,
        )

        severity_plan = memory_dataset(
            f"{TASK_ID}-severity", concrete_events
        ).sem_map(
            cols=[
                {
                    "name": "complaint_severity",
                    "type": str,
                    "desc": (
                        "Exactly one of minor_inconvenience, "
                        "moderate_safety_event, or severe_safety_event."
                    ),
                }
            ],
            desc="Classify the complaint severity.",
            depends_on=["summary_text"],
        )
        started = time.time()
        severity_result = severity_plan.run(config)
        severity_labeled = result_frame(
            severity_result,
            concrete_events,
            ["complaint_severity"],
        )
        severity_labeled["complaint_severity"] = severity_labeled[
            "complaint_severity"
        ].map(lambda value: normalize_enum(value, COMPLAINT_SEVERITIES))
        if severity_labeled["complaint_severity"].isna().any():
            raise ValueError(f"{TASK_ID}: invalid complaint severity label")
        tracker.record_semantic(
            "sem_map",
            len(concrete_events),
            severity_labeled,
            severity_result,
            time.time() - started,
        )

        severity_projection = severity_labeled[
            [
                "complaint_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "summary_text",
                "campaign_number",
                "recall_corrective_action",
                "complaint_severity",
            ]
        ].copy()
        tracker.record("project", len(severity_labeled), severity_projection)

        remedy_plan = memory_dataset(
            f"{TASK_ID}-remedy", severity_projection
        ).sem_map(
            cols=[
                {
                    "name": "recall_remedy_style",
                    "type": str,
                    "desc": (
                        "Exactly one of software_update, part_replacement, or other."
                    ),
                }
            ],
            desc="Classify the recall remedy style.",
            depends_on=["recall_corrective_action"],
        )
        started = time.time()
        remedy_result = remedy_plan.run(config)
        remedy_labeled = result_frame(
            remedy_result,
            severity_projection,
            ["recall_remedy_style"],
        )
        remedy_labeled["recall_remedy_style"] = remedy_labeled[
            "recall_remedy_style"
        ].map(lambda value: normalize_enum(value, REMEDY_STYLES) or "other")
        tracker.record_semantic(
            "sem_map",
            len(severity_projection),
            remedy_labeled,
            remedy_result,
            time.time() - started,
        )

        aggregation_input = remedy_labeled[
            [
                "complaint_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "campaign_number",
                "complaint_severity",
                "recall_remedy_style",
            ]
        ].copy()
        tracker.record("project", len(remedy_labeled), aggregation_input)

        aggregation_input["severe_indicator"] = (
            aggregation_input["crash_flag"].map(parse_bool)
            | pd.to_numeric(
                aggregation_input["injury_count"],
                errors="coerce",
            ).fillna(0).gt(0)
            | aggregation_input["vehicle_towed_flag"].map(parse_bool)
        )
        if aggregation_input.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "severe_indicator_rate",
                    "dominant_complaint_severity",
                    "dominant_recall_remedy_style",
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
                    severe_indicator_rate=("severe_indicator", "mean"),
                    dominant_complaint_severity=(
                        "complaint_severity",
                        stable_mode,
                    ),
                    dominant_recall_remedy_style=(
                        "recall_remedy_style",
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

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_five)

        top_five.insert(0, "rank", range(1, len(top_five) + 1))
        result = top_five[
            [
                "rank",
                "campaign_number",
                "matched_complaint_count",
                "severe_indicator_rate",
                "dominant_complaint_severity",
                "dominant_recall_remedy_style",
            ]
        ].copy()
        tracker.record("project", len(top_five), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
