#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-013."""

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
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    parse_bool,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "aviation_safety-013"
SYMPTOM_CATEGORIES = (
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
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem", "result_summary"],
        )
        tracker.record("scan", None, incidents)

        aircraft = incidents.loc[
            incidents["primary_problem"] == "Aircraft"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), aircraft)

        reports = load_selected_texts("asrs", aircraft)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record("scan", len(aircraft), reports)

        extraction_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "qualifies",
                    "type": bool,
                    "desc": (
                        "True only when the report describes an engine thrust, "
                        "power, oil, temperature, vibration, flameout, or other "
                        "engine-indication problem."
                    ),
                },
                {
                    "name": "symptom_category",
                    "type": str,
                    "desc": (
                        "For a qualifying report, select the dominant symptom as "
                        "exactly one of thrust_loss, oil_indication, temperature, "
                        "vibration, flameout, or other_engine_indication."
                    ),
                },
                {
                    "name": "immediate_crew_action",
                    "type": str,
                    "desc": (
                        "The crew's immediate response to the engine symptom, in "
                        "no more than six words."
                    ),
                },
                {
                    "name": "primary_outcome",
                    "type": str,
                    "desc": (
                        "Exactly one of emergency, diversion, return, "
                        "continued_flight, or maintenance_only."
                    ),
                },
            ],
            desc=(
                "For qualifying engine-problem reports, extract one dominant "
                "symptom category, the immediate crew action, and the primary "
                "outcome. Do not qualify non-engine equipment problems."
            ),
            depends_on=["incident_id", "result_summary", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result)
        extracted["qualifies"] = extracted["qualifies"].map(parse_bool)
        extracted["symptom_category"] = extracted[
            "symptom_category"
        ].map(lambda value: normalize_enum(value, SYMPTOM_CATEGORIES))
        extracted["primary_outcome"] = extracted["primary_outcome"].map(
            lambda value: normalize_enum(value, OUTCOMES)
        )
        extracted["immediate_crew_action"] = (
            extracted["immediate_crew_action"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            extraction_result,
            time.time() - started,
        )

        selected = extracted.loc[
            extracted["qualifies"]
            & extracted["symptom_category"].notna()
            & extracted["primary_outcome"].notna()
            & extracted["immediate_crew_action"].ne(""),
            [
                "incident_id",
                "symptom_category",
                "immediate_crew_action",
                "primary_outcome",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), selected)

        grouped = (
            selected.groupby("symptom_category", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_immediate_action=(
                    "immediate_crew_action",
                    stable_mode,
                ),
                most_common_outcome=("primary_outcome", stable_mode),
            )
            .reset_index()
        )
        tracker.record("groupby", len(selected), grouped)

        ordered = grouped.sort_values(
            ["incident_count", "symptom_category"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("sort", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
