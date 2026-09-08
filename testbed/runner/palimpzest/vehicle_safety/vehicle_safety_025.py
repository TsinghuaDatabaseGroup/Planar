#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-025."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_jsonl,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-025"
DATASET = "nhtsa_vehicle_safety"


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

        adas_complaints = complaints.loc[
            complaints["component_id"] == "ADAS"
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), adas_complaints)

        complaint_plan = memory_dataset(
            f"{TASK_ID}-complaints", adas_complaints
        ).sem_filter(
            filter=(
                "The complaint summary specifically describes phantom braking: "
                "sudden, automatic, unprovoked braking by a driver-assist system."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        complaint_result = complaint_plan.run(config)
        phantom_braking = result_frame(complaint_result, adas_complaints)
        tracker.record_semantic(
            "sem_filter",
            len(adas_complaints),
            phantom_braking,
            complaint_result,
            time.time() - started,
        )

        complaint_counts = (
            phantom_braking.groupby("make", as_index=False, dropna=False)
            .agg(
                phantom_braking_complaint_count=(
                    "complaint_id",
                    "nunique",
                )
            )
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(phantom_braking), complaint_counts)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(columns={"vehicle_make": "make"})
        tracker.record("scan", None, recalls)

        adas_recalls = recalls.loc[
            recalls["component_component_id"] == "ADAS"
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), adas_recalls)

        recall_plan = memory_dataset(
            f"{TASK_ID}-recalls", adas_recalls
        ).sem_filter(
            filter=(
                "The recall defect summary discusses automatic braking, "
                "driver-assist braking, adaptive cruise braking, or speed-control "
                "behavior as the recall defect."
            ),
            depends_on=["risk_defect_summary"],
        )
        started = time.time()
        recall_result = recall_plan.run(config)
        matching_recalls = result_frame(recall_result, adas_recalls)
        tracker.record_semantic(
            "sem_filter",
            len(adas_recalls),
            matching_recalls,
            recall_result,
            time.time() - started,
        )

        recall_counts = (
            matching_recalls.groupby("make", as_index=False, dropna=False)
            .agg(matching_adas_recall_count=("campaign_number", "nunique"))
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(matching_recalls), recall_counts)

        joined = complaint_counts.merge(recall_counts, on="make", how="inner")
        tracker.record(
            "join",
            {"left": len(complaint_counts), "right": len(recall_counts)},
            joined,
        )

        ordered = joined.sort_values(
            ["phantom_braking_complaint_count", "make"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(joined), ordered)

        top_four = ordered.head(4).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_four)

        top_four.insert(0, "rank", range(1, len(top_four) + 1))
        result = top_four[
            [
                "rank",
                "make",
                "phantom_braking_complaint_count",
                "matching_adas_recall_count",
            ]
        ].copy()
        tracker.record("project", len(top_four), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
