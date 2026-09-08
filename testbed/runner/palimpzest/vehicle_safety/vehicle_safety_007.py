#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-007."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-007"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["make", "model", "injury_count", "fire_flag", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        filtered = complaints.loc[
            (complaints["make"] == "HYUNDAI")
            & (complaints["model"] == "ELANTRA")
            & complaints["injury_count"].ge(1)
            & ~complaints["fire_flag"]
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), filtered)

        semantic_plan = memory_dataset(TASK_ID, filtered).sem_filter(
            filter=(
                "The complaint summary describes a loss of vehicle control "
                "while driving."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        semantic_result = semantic_plan.run(config)
        matching = result_frame(semantic_result, filtered)
        tracker.record_semantic(
            "sem_filter",
            len(filtered),
            matching,
            semantic_result,
            time.time() - started,
        )

        answer = int(len(matching))
        tracker.record("groupby", len(matching), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
