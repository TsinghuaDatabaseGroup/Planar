#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-012."""

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

TASK_ID = "vehicle_safety-012"
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

        selected_makes = complaints.loc[
            complaints["make"].isin(["KIA", "HYUNDAI"])
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), selected_makes)

        relevance_plan = memory_dataset(TASK_ID, selected_makes).sem_filter(
            filter=(
                "The complaint summary describes fire, smoke, a burning smell, "
                "melting, or an electrical short related to the vehicle system."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        hazardous_complaints = result_frame(relevance_result, selected_makes)
        tracker.record_semantic(
            "sem_filter",
            len(selected_makes),
            hazardous_complaints,
            relevance_result,
            time.time() - started,
        )

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_park_outside",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_make": "make",
                "component_component_id": "component_id",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("scan", None, recalls)

        park_outside = recalls.loc[
            recalls["make"].isin(["KIA", "HYUNDAI"])
            & recalls["risk_park_outside"].eq("TRUE")
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), park_outside)

        deduplicated = park_outside.drop_duplicates(
            subset=["campaign_number"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(park_outside), deduplicated)

        joined = hazardous_complaints.merge(
            deduplicated[
                ["campaign_number", "make", "component_id", "defect_summary"]
            ],
            on=["make", "component_id"],
            how="inner",
        )
        tracker.record(
            "join",
            {"left": len(hazardous_complaints), "right": len(deduplicated)},
            joined,
        )

        grouped = (
            joined.groupby(
                ["campaign_number", "make", "component_id"],
                as_index=False,
                dropna=False,
            )
            .agg(
                matched_complaint_count=("complaint_id", "nunique"),
                defect_summary=("defect_summary", "first"),
            )
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(joined), grouped)

        result = grouped[
            [
                "campaign_number",
                "make",
                "component_id",
                "matched_complaint_count",
                "defect_summary",
            ]
        ].copy()
        tracker.record("project", len(grouped), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
