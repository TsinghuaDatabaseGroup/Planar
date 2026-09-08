#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-003."""

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
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-003"
DATASET = "nhtsa_vehicle_safety"


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
                "model_year",
                "component_id",
                "injury_count",
                "fire_flag",
                "summary_text",
            ],
        )
        tracker.record("scan", None, complaints)

        filtered = complaints.loc[
            (complaints["make"] == "DODGE")
            & (complaints["component_id"] == "AIRBAG")
            & complaints["injury_count"].ge(1)
            & ~complaints["fire_flag"]
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), filtered)

        extraction_plan = memory_dataset(TASK_ID, filtered).sem_map(
            cols=[
                {
                    "name": "airbag_failure_mode",
                    "type": str,
                    "desc": (
                        "The primary airbag failure mode as a short canonical "
                        "phrase, such as non-deployment, unintended deployment, "
                        "inflator rupture, occupant burn or laceration, "
                        "warning-light fault, or other."
                    ),
                }
            ],
            desc="Extract the primary airbag failure mode described in the complaint.",
            depends_on=["summary_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            filtered,
            ["airbag_failure_mode"],
        )
        tracker.record_semantic(
            "sem_map",
            len(filtered),
            extracted,
            extraction_result,
            time.time() - started,
        )

        result = extracted[
            ["complaint_id", "model_year", "airbag_failure_mode"]
        ].copy()
        tracker.record("project", len(extracted), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
