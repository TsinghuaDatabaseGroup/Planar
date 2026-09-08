#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for aviation_safety-024."""

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
    load_selected_texts,
    load_table,
    memory_dataset,
    result_frame,
    run_optimized_plan,
    save_output,
)

TASK_ID = "aviation_safety-024"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
SOURCE_COLUMNS = [
    "incident_id",
    "incident_date",
    "state_reference",
    "locale_reference_type",
    "primary_problem",
    "text_file",
]
JOIN_COLUMNS = [
    "left_incident_id",
    "left_incident_date",
    "left_state_reference",
    "left_locale_reference_type",
    "left_primary_problem",
    "left_text",
    "right_incident_id",
    "right_incident_date",
    "right_state_reference",
    "right_locale_reference_type",
    "right_primary_problem",
    "right_text",
]


def main() -> None:
    config = get_config(
        max_tokens=4096,
        all_optimizations=True,
        include_small_model=True,
    )
    tracker = StepTracker(TASK_ID, optimizer_strategy=OPTIMIZER)

    with Timer() as timer:
        left_source = load_table("asrs", "incidents.csv", SOURCE_COLUMNS)
        tracker.record("scan", None, left_source)
        left_reports = load_selected_texts("asrs", left_source).rename(
            columns={
                "incident_id": "left_incident_id",
                "incident_date": "left_incident_date",
                "state_reference": "left_state_reference",
                "locale_reference_type": "left_locale_reference_type",
                "primary_problem": "left_primary_problem",
                "text": "left_text",
            }
        )[
            [
                "left_incident_id",
                "left_incident_date",
                "left_state_reference",
                "left_locale_reference_type",
                "left_primary_problem",
                "left_text",
            ]
        ]
        tracker.record("scan", len(left_source), left_reports)

        right_source = load_table("asrs", "incidents.csv", SOURCE_COLUMNS)
        tracker.record("scan", None, right_source)
        right_reports = load_selected_texts("asrs", right_source).rename(
            columns={
                "incident_id": "right_incident_id",
                "incident_date": "right_incident_date",
                "state_reference": "right_state_reference",
                "locale_reference_type": "right_locale_reference_type",
                "primary_problem": "right_primary_problem",
                "text": "right_text",
            }
        )[
            [
                "right_incident_id",
                "right_incident_date",
                "right_state_reference",
                "right_locale_reference_type",
                "right_primary_problem",
                "right_text",
            ]
        ]
        tracker.record("scan", len(right_source), right_reports)

        plan = memory_dataset(
            f"{TASK_ID}-left",
            left_reports,
        ).sem_join(
            memory_dataset(f"{TASK_ID}-right", right_reports),
            condition=(
                "Evaluate only pairs where left_incident_id is smaller than "
                "right_incident_id and left_incident_date equals "
                "right_incident_date, left_state_reference equals "
                "right_state_reference, left_locale_reference_type equals "
                "right_locale_reference_type, and left_primary_problem equals "
                "right_primary_problem. Keep a pair only when both narratives "
                "describe the same underlying occurrence, requiring a compatible "
                "distinctive event progression, measurements, actions, and "
                "outcome; generic topical similarity is insufficient."
            ),
            depends_on=JOIN_COLUMNS,
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        matched = result_frame(semantic_result)
        if matched.empty:
            matched = pd.DataFrame(columns=JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(left_reports), "right": len(right_reports)},
            matched,
            semantic_result,
            time.time() - started,
        )

        projected = matched[["left_incident_date", "left_incident_id", "right_incident_id"]].rename(
            columns={"left_incident_date": "incident_date"}
        )
        tracker.record("project", len(matched), projected)
        ordered = projected.sort_values(
            ["incident_date", "left_incident_id", "right_incident_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        tracker.record("order_by", len(projected), ordered)
        ordered.insert(0, "rank", range(1, len(ordered) + 1))
        answer_frame = ordered[["rank", "incident_date", "left_incident_id", "right_incident_id"]]
        tracker.record("project", len(ordered), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
