#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-027."""

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
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-027"
DECISION_TYPES = (
    "diversion",
    "airport_change",
    "holding",
    "approach_change",
    "altitude_change",
    "route_change",
)
CLASSIFICATION_LABELS = (*DECISION_TYPES, "not_target")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "local_time_of_day"],
        )
        tracker.record("scan", None, incidents)

        afternoon = incidents.loc[
            incidents["local_time_of_day"] == "1201-1800"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), afternoon)

        reports = load_selected_texts("asrs", afternoon)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(afternoon), reports)

        weather_plan = memory_dataset(
            f"{TASK_ID}-weather", reports
        ).sem_filter(
            (
                "Keep this report only if weather or visibility was the primary "
                "cause of an operational decision change, rather than incidental "
                "background."
            ),
            depends_on=["text"],
        )
        started = time.time()
        weather_result = weather_plan.run(config)
        weather_caused = result_frame(weather_result)[
            ["incident_id", "text"]
        ]
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            weather_caused,
            weather_result,
            time.time() - started,
        )

        operational_plan = memory_dataset(
            f"{TASK_ID}-operational", weather_caused
        ).sem_filter(
            (
                "Keep this report unless turbulence affected only passenger "
                "comfort without changing the flight's operation."
            ),
            depends_on=["text"],
        )
        started = time.time()
        operational_result = operational_plan.run(config)
        operational = result_frame(operational_result)[
            ["incident_id", "text"]
        ]
        tracker.record_semantic(
            "sem_filter",
            len(weather_caused),
            operational,
            operational_result,
            time.time() - started,
        )

        classification_plan = memory_dataset(
            f"{TASK_ID}-classification", operational
        ).sem_map(
            cols=[
                {
                    "name": "decision_change_type",
                    "type": str,
                    "desc": (
                        "Exactly one of diversion, airport_change, holding, "
                        "approach_change, altitude_change, route_change, or "
                        "not_target. Select the first applicable category in "
                        "that order; use not_target if none applies."
                    ),
                }
            ],
            desc=(
                "Classify the weather-caused operational decision change and "
                "apply this precedence when several occurred: diversion, "
                "airport_change, holding, approach_change, altitude_change, "
                "route_change."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(classification_result)
        classified["decision_change_type"] = classified[
            "decision_change_type"
        ].map(lambda value: normalize_enum(value, CLASSIFICATION_LABELS))
        tracker.record_semantic(
            "sem_map",
            len(operational),
            classified,
            classification_result,
            time.time() - started,
        )

        selected = classified.loc[
            classified["decision_change_type"].isin(DECISION_TYPES),
            ["incident_id", "decision_change_type"],
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), selected)

        grouped = (
            selected.groupby("decision_change_type", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        decision_order = {
            label: index for index, label in enumerate(DECISION_TYPES)
        }
        grouped = grouped.sort_values(
            "decision_change_type",
            key=lambda values: values.map(decision_order),
        ).reset_index(drop=True)
        tracker.record("groupby", len(selected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
