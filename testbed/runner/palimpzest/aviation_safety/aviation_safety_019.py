#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for aviation_safety-019."""

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
    parse_bool,
    result_frame,
    run_optimized_plan,
    save_output,
)

TASK_ID = "aviation_safety-019"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
CONSEQUENCES = (
    "emergency_declaration",
    "diversion",
    "return",
    "holding",
    "continued_with_concern",
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
            ["incident_id", "text_file", "far_part", "mission", "result_summary"],
        )
        tracker.record("scan", None, incidents)

        part121_passenger = incidents.loc[
            (incidents["far_part"] == "Part 121") & (incidents["mission"] == "Passenger")
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), part121_passenger)

        reports = load_selected_texts("asrs", part121_passenger)[["incident_id", "result_summary", "text"]]
        tracker.record("scan", len(part121_passenger), reports)

        plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "operational_consequence",
                    "type": str,
                    "desc": (
                        "The highest applicable consequence using exactly this "
                        "precedence: emergency_declaration, diversion, return, "
                        "holding, continued_with_concern; otherwise null."
                    ),
                },
                {
                    "name": "has_numeric_fuel_value",
                    "type": bool,
                    "desc": (
                        "Whether the qualifying report states a numeric fuel quantity, reserve, or calculation value."
                    ),
                },
                {
                    "name": "has_numeric_time_value",
                    "type": bool,
                    "desc": ("Whether the qualifying report states a numeric time value related to the fuel decision."),
                },
            ],
            desc=(
                "Extract one row when a fuel quantity, reserve, calculation, or "
                "dispatch-release decision directly affected the operation, "
                "assigning its highest-precedence operational consequence and "
                "the two independent numeric-value indicators."
            ),
            depends_on=["incident_id", "result_summary", "text"],
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        extracted = result_frame(
            semantic_result,
            reports,
            [
                "operational_consequence",
                "has_numeric_fuel_value",
                "has_numeric_time_value",
            ],
        )
        extracted["operational_consequence"] = extracted["operational_consequence"].map(
            lambda value: normalize_enum(value, CONSEQUENCES)
        )
        extracted["has_numeric_fuel_value"] = extracted["has_numeric_fuel_value"].map(parse_bool)
        extracted["has_numeric_time_value"] = extracted["has_numeric_time_value"].map(parse_bool)
        qualified = (
            extracted.loc[
                extracted["operational_consequence"].notna(),
                [
                    "incident_id",
                    "operational_consequence",
                    "has_numeric_fuel_value",
                    "has_numeric_time_value",
                ],
            ]
            .drop_duplicates(["incident_id"])
            .reset_index(drop=True)
        )
        tracker.record_semantic(
            "sem_map",
            len(reports),
            qualified,
            semantic_result,
            time.time() - started,
        )

        rows = []
        for consequence in CONSEQUENCES:
            subset = qualified.loc[qualified["operational_consequence"] == consequence]
            if subset.empty:
                continue
            rows.append(
                {
                    "operational_consequence": consequence,
                    "incident_count": int(subset["incident_id"].nunique()),
                    "with_numeric_fuel_value_count": int(
                        subset.loc[subset["has_numeric_fuel_value"], "incident_id"].nunique()
                    ),
                    "with_numeric_time_value_count": int(
                        subset.loc[subset["has_numeric_time_value"], "incident_id"].nunique()
                    ),
                }
            )
        grouped = pd.DataFrame.from_records(
            rows,
            columns=[
                "operational_consequence",
                "incident_count",
                "with_numeric_fuel_value_count",
                "with_numeric_time_value_count",
            ],
        )
        tracker.record("groupby", len(qualified), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
