#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for aviation_safety-020."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

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

TASK_ID = "aviation_safety-020"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
DISRUPTION_REASONS = (
    "predeparture_component_discrepancy",
    "mel_or_deferred_item",
    "maintenance_documentation_or_release",
    "inflight_component_or_system_problem",
    "postflight_inspection_or_repair",
    "not_target",
)


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
            "inputs/OP-SEM_GROUP_BY-003.jsonl",
        )[["incident_id", "result_labels", "contributing_labels", "text"]]
        tracker.record("scan", None, reports)

        plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "disruption_reason",
                    "type": str,
                    "desc": (
                        "Exactly one of predeparture_component_discrepancy, "
                        "mel_or_deferred_item, "
                        "maintenance_documentation_or_release, "
                        "inflight_component_or_system_problem, "
                        "postflight_inspection_or_repair, or not_target when no "
                        "category clearly fits."
                    ),
                }
            ],
            desc=(
                "Assign exactly one primary disruption reason supported by the "
                "structured labels and incident narrative."
            ),
            depends_on=[
                "incident_id",
                "result_labels",
                "contributing_labels",
                "text",
            ],
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        classified = result_frame(
            semantic_result,
            reports,
            ["disruption_reason"],
        )
        classified["disruption_reason"] = classified["disruption_reason"].map(
            lambda value: normalize_enum(value, DISRUPTION_REASONS)
        )
        classified["disruption_reason"] = classified["disruption_reason"].fillna("not_target")
        tracker.record_semantic(
            "sem_map",
            len(reports),
            classified,
            semantic_result,
            time.time() - started,
        )

        unique_incidents = classified[["incident_id", "disruption_reason"]].drop_duplicates(["incident_id"])

        rows = []
        for label in DISRUPTION_REASONS:
            subset = unique_incidents.loc[unique_incidents["disruption_reason"] == label]
            if subset.empty:
                continue
            rows.append(
                {
                    "disruption_reason": label,
                    "incident_count": int(subset["incident_id"].nunique()),
                }
            )
        grouped = pd.DataFrame.from_records(
            rows,
            columns=["disruption_reason", "incident_count"],
        )
        tracker.record("groupby", len(unique_incidents), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
