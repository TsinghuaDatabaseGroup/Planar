#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-010."""

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
)

TASK_ID = "aviation_safety-010"
CABIN_SOURCES = (
    "medical_event",
    "security_issue",
    "policy_issue",
    "door_issue",
    "slide_issue",
    "cabin_equipment_issue",
    "passenger_behavior",
    "boarding_issue",
    "deplaning_issue",
    "flight_attendant_communication",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem"],
        )
        tracker.record("scan", None, incidents)

        filtered = incidents.loc[
            incidents["primary_problem"] == "Human Factors"
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(incidents),
            filtered,
        )

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "scan",
            len(filtered),
            reports,
        )

        semantic = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "qualifies",
                    "type": bool,
                    "desc": (
                        "True only when at least one cabin-side issue in the permitted "
                        "categories affected a flight-deck decision."
                    ),
                },
                {
                    "name": "cabin_side_source",
                    "type": str,
                    "desc": (
                        "When qualifies is true, select exactly one source using this "
                        "precedence: medical_event, security_issue, policy_issue, "
                        "door_issue, slide_issue, cabin_equipment_issue, "
                        "passenger_behavior, boarding_issue, deplaning_issue, "
                        "flight_attendant_communication. Otherwise use not_applicable."
                    ),
                },
            ],
            desc=(
                "Determine whether a cabin-side issue affected a flight-deck decision. "
                "If several permitted source categories apply, assign the first one "
                "in the stated precedence order; omit issues that did not affect such "
                "a decision."
            ),
            depends_on=["text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        raw = result_frame(semantic_result)
        raw["qualifies"] = raw["qualifies"].map(parse_bool)
        raw["cabin_side_source"] = raw["cabin_side_source"].map(
            lambda value: normalize_enum(value, CABIN_SOURCES)
        )
        classified = raw.loc[
            raw["qualifies"] & raw["cabin_side_source"].notna(),
            ["incident_id", "cabin_side_source"],
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(reports),
            classified,
            semantic_result,
            time.time() - started,
        )

        grouped = (
            classified.groupby("cabin_side_source", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        source_order = {
            label: index for index, label in enumerate(CABIN_SOURCES)
        }
        grouped = grouped.sort_values(
            "cabin_side_source",
            key=lambda values: values.map(source_order),
        ).reset_index(drop=True)
        tracker.record(
            "groupby",
            len(classified),
            grouped,
        )
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
