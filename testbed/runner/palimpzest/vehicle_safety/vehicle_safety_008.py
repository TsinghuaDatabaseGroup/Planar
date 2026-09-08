#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-008."""

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
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-008"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        recalls = load_jsonl(DATASET, "recalls.jsonl")
        tracker.record("scan", None, recalls)

        filtered = recalls.loc[
            (recalls["vehicle_make"] == "TESLA")
            & (recalls["component_component_id"] == "STEERING")
            & (recalls["risk_park_outside"] == "FALSE")
            & (recalls["risk_do_not_drive"] == "FALSE"),
            ["campaign_number", "risk_defect_summary"],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), filtered)

        deduplicated = filtered.drop_duplicates(
            subset=["campaign_number"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(filtered), deduplicated)

        extraction_plan = memory_dataset(TASK_ID, deduplicated).sem_map(
            cols=[
                {
                    "name": "primary_cause_phrase",
                    "type": str,
                    "desc": (
                        "The primary failure cause, written as a short canonical "
                        "phrase of at most six words."
                    ),
                }
            ],
            desc="Identify the primary failure cause described in the defect summary.",
            depends_on=["risk_defect_summary"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            deduplicated,
            ["primary_cause_phrase"],
        )
        tracker.record_semantic(
            "sem_map",
            len(deduplicated),
            extracted,
            extraction_result,
            time.time() - started,
        )

        result = extracted[["campaign_number", "primary_cause_phrase"]].copy()
        tracker.record("project", len(extracted), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
