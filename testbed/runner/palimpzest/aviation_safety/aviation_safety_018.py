#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for aviation_safety-018."""

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
    normalize_enum,
    result_frame,
    run_optimized_plan,
    save_output,
    stable_mode,
)

TASK_ID = "aviation_safety-018"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
SYMPTOMS = (
    "thrust_loss",
    "oil_indication",
    "temperature",
    "vibration",
    "flameout",
    "other_engine_indication",
)
OUTCOMES = (
    "emergency",
    "diversion",
    "return",
    "continued_flight",
    "maintenance_only",
)


def main() -> None:
    config = get_config(
        max_tokens=4096,
        all_optimizations=True,
        include_small_model=True,
    )
    tracker = StepTracker(TASK_ID, optimizer_strategy=OPTIMIZER)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem", "result_summary"],
        )
        tracker.record("scan", None, incidents)

        aircraft = incidents.loc[incidents["primary_problem"] == "Aircraft"].reset_index(drop=True)
        tracker.record("filter", len(incidents), aircraft)

        reports = load_selected_texts("asrs", aircraft)[["incident_id", "result_summary", "text"]]
        tracker.record("scan", len(aircraft), reports)

        plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "symptom_category",
                    "type": str,
                    "desc": (
                        "Exactly one of thrust_loss, oil_indication, temperature, "
                        "vibration, flameout, or other_engine_indication for a "
                        "qualifying engine problem; otherwise null."
                    ),
                },
                {
                    "name": "primary_outcome",
                    "type": str,
                    "desc": (
                        "Exactly one of emergency, diversion, return, "
                        "continued_flight, or maintenance_only for the qualifying "
                        "engine problem; otherwise null."
                    ),
                },
            ],
            desc=(
                "Extract one row only for an engine thrust, power, oil, "
                "temperature, vibration, or flameout problem, assigning its "
                "symptom category and primary outcome."
            ),
            depends_on=["incident_id", "result_summary", "text"],
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        extracted = result_frame(
            semantic_result,
            reports,
            ["symptom_category", "primary_outcome"],
        )
        extracted["symptom_category"] = extracted["symptom_category"].map(lambda value: normalize_enum(value, SYMPTOMS))
        extracted["primary_outcome"] = extracted["primary_outcome"].map(lambda value: normalize_enum(value, OUTCOMES))
        qualified = extracted.loc[
            extracted["symptom_category"].notna() & extracted["primary_outcome"].notna(),
            ["incident_id", "symptom_category", "primary_outcome"],
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(reports),
            qualified,
            semantic_result,
            time.time() - started,
        )

        rows = []
        for symptom in SYMPTOMS:
            subset = qualified.loc[qualified["symptom_category"] == symptom]
            if subset.empty:
                continue
            rows.append(
                {
                    "symptom_category": symptom,
                    "incident_count": int(subset["incident_id"].nunique()),
                    "most_common_outcome": stable_mode(subset["primary_outcome"]),
                }
            )
        grouped = (
            pd.DataFrame.from_records(
                rows,
                columns=[
                    "symptom_category",
                    "incident_count",
                    "most_common_outcome",
                ],
            )
            .sort_values(
                ["incident_count", "symptom_category"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(qualified), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
