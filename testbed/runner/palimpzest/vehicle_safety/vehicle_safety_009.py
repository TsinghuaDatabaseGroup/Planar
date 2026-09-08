#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-009."""

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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-009"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["make", "model", "model_year", "crash_flag", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        filtered = complaints.loc[
            (complaints["make"] == "SUBARU")
            & (complaints["model"] == "OUTBACK")
            & complaints["model_year"].ge(2024)
            & ~complaints["crash_flag"],
            ["summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), filtered)

        aggregate_plan = memory_dataset(TASK_ID, filtered).sem_agg(
            col={
                "name": "dominant_mode",
                "type": str,
                "desc": "Exactly electrical, mechanical, software, or mixed.",
            },
            agg=(
                "Determine the dominant failure mode across all input complaint "
                "summaries: electrical, mechanical, or software. If the dominant "
                "mode is unclear or tied, return mixed."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        aggregate_result = aggregate_plan.run(config)
        aggregate_frame = result_frame(aggregate_result)
        tracker.record_semantic(
            "sem_agg",
            len(filtered),
            aggregate_frame,
            aggregate_result,
            time.time() - started,
        )

        if aggregate_frame.empty:
            raise ValueError(f"{TASK_ID}: semantic aggregate returned no answer")
        answer = normalize_enum(
            aggregate_frame.iloc[0]["dominant_mode"],
            ("electrical", "mechanical", "software", "mixed"),
        )
        if answer is None:
            raise ValueError(f"{TASK_ID}: semantic aggregate returned invalid label")

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
