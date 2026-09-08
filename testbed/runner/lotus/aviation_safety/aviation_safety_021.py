#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-021."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_jsonl,
    normalize_enum,
    save_output,
    setup,
)


TASK_ID = "aviation_safety-021"
BARRIER_STATUSES = ("successful_barrier", "not_fully_supported")


def main():
    setup(max_tokens=256)
    tracker = StepTracker()

    with Timer() as timer:
        candidates = load_jsonl(
            "operator_implement",
            "inputs/NASA_ASRS-030_safety_barrier_candidates.jsonl",
        )[["incident_id", "text"]].copy()
        tracker.record(
            "SCAN_TABLE(operator_implement/inputs/"
            "NASA_ASRS-030_safety_barrier_candidates.jsonl)",
            None,
            len(candidates),
            output=candidates,
        )

        with tracker.step(
            "SEM_CLASSIFY(safety barrier status)",
            input_rows=len(candidates),
        ) as step:
            classified = candidates.sem_map(
                "Assign exactly one barrier_status to narrative {text}. Use "
                "successful_barrier only when the narrative clearly establishes "
                "hazard detection, a concrete intervention, and either avoidance "
                "of a worse consequence or stabilization of the situation; "
                "otherwise use not_fully_supported. Output only the label.",
                suffix="barrier_status",
            )
            classified["barrier_status"] = classified["barrier_status"].map(
                lambda value: normalize_enum(value, BARRIER_STATUSES)
                or "not_fully_supported"
            )
            step.set_output(classified)

        grouped = (
            classified.groupby("barrier_status", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        status_order = {
            label: index for index, label in enumerate(BARRIER_STATUSES)
        }
        grouped = grouped.sort_values(
            "barrier_status",
            key=lambda values: values.map(status_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY(barrier_status, COUNT_DISTINCT(incident_id))",
            len(classified),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
