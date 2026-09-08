#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for aviation_safety-021."""

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
    normalize_enum,
    result_frame,
    run_optimized_plan,
    save_output,
)

TASK_ID = "aviation_safety-021"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
BARRIER_LABELS = ("successful_barrier", "not_fully_supported")


def main() -> None:
    config = get_config(
        max_tokens=4096,
        all_optimizations=True,
        include_small_model=True,
    )
    tracker = StepTracker(TASK_ID, optimizer_strategy=OPTIMIZER)

    with Timer() as timer:
        reports = load_jsonl(
            "operator_implement",
            "inputs/NASA_ASRS-030_safety_barrier_candidates.jsonl",
        )[["incident_id", "text"]]
        tracker.record("scan", None, reports)

        plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "barrier_status",
                    "type": str,
                    "desc": (
                        "successful_barrier only when the narrative clearly "
                        "establishes hazard detection, a concrete intervention, "
                        "and avoidance of a worse consequence or stabilization "
                        "of the situation; otherwise "
                        "not_fully_supported."
                    ),
                }
            ],
            desc=("Assign exactly one barrier status to every candidate incident."),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        classified = result_frame(
            semantic_result,
            reports,
            ["barrier_status"],
        )
        classified["barrier_status"] = classified["barrier_status"].map(
            lambda value: normalize_enum(value, BARRIER_LABELS)
        )
        classified["barrier_status"] = classified["barrier_status"].fillna("not_fully_supported")
        tracker.record_semantic(
            "sem_map",
            len(reports),
            classified,
            semantic_result,
            time.time() - started,
        )

        unique_incidents = classified.drop_duplicates(["incident_id"]).reset_index(drop=True)
        grouped = (
            unique_incidents.groupby("barrier_status", as_index=False)["incident_id"]
            .nunique()
            .rename(columns={"incident_id": "incident_count"})
        )
        label_order = {label: index for index, label in enumerate(BARRIER_LABELS)}
        grouped = (
            grouped.assign(_label_order=grouped["barrier_status"].map(label_order))
            .sort_values("_label_order")
            .drop(columns="_label_order")
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(unique_incidents), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
