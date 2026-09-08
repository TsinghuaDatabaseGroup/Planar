#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-029."""

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

TASK_ID = "vehicle_safety-029"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["make", "component_id", "injury_count", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        selected = complaints.loc[
            (complaints["make"] == "HONDA")
            & complaints["component_id"].isin(["ADAS", "BRAKES"])
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), selected)

        tangible_plan = memory_dataset(
            f"{TASK_ID}-tangible", selected
        ).sem_filter(
            filter=(
                "The complaint summary describes a tangible safety event, not a "
                "minor inconvenience such as screen flicker, infotainment lag, or "
                "a cosmetic blemish."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        tangible_result = tangible_plan.run(config)
        tangible = result_frame(tangible_result, selected)
        tracker.record_semantic(
            "sem_filter",
            len(selected),
            tangible,
            tangible_result,
            time.time() - started,
        )

        motion_plan = memory_dataset(
            f"{TASK_ID}-motion", tangible
        ).sem_filter(
            filter=(
                "The complaint summary indicates that the problem manifested "
                "while the vehicle was being driven, not while it was parked or "
                "stationary."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        motion_result = motion_plan.run(config)
        in_motion = result_frame(motion_result, tangible)
        tracker.record_semantic(
            "sem_filter",
            len(tangible),
            in_motion,
            motion_result,
            time.time() - started,
        )

        grouped = (
            in_motion.groupby("component_id", as_index=False, dropna=False)
            .agg(avg_injury=("injury_count", "mean"))
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(in_motion), grouped)

        averages = dict(
            zip(
                grouped["component_id"],
                grouped["avg_injury"],
                strict=True,
            )
        )
        adas_average = float(averages.get("ADAS", float("nan")))
        brakes_average = float(averages.get("BRAKES", float("nan")))
        if adas_average - brakes_average > 0.01:
            answer = "ADAS_higher"
        elif brakes_average - adas_average > 0.01:
            answer = "BRAKES_higher"
        else:
            answer = "tie"
        tracker.record("project", len(grouped), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
