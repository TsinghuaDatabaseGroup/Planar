#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-001."""

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
    normalize_free_label,
    parse_bool,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "aviation_safety-001"
TRIGGERS = (
    "unstable_approach",
    "traffic_spacing",
    "runway_occupancy",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "far_part", "mission"],
        )
        tracker.record("scan", None, incidents)

        filtered = incidents.loc[
            (incidents["far_part"] == "Part 121")
            & (incidents["mission"] == "Passenger")
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
                        "True only when the reporting aircraft actually performed "
                        "a go-around or missed approach whose primary trigger was "
                        "an unstable approach, traffic spacing, or runway occupancy, "
                        "and aircraft equipment failure was not the primary trigger."
                    ),
                },
                {
                    "name": "primary_trigger",
                    "type": str,
                    "desc": (
                        "When qualifies is true, exactly one of "
                        "unstable_approach, traffic_spacing, runway_occupancy; "
                        "otherwise not_applicable."
                    ),
                },
                {
                    "name": "initiating_role",
                    "type": str,
                    "desc": (
                        "When qualifies is true, the crew or ATC role that initiated "
                        "the maneuver, expressed as a concise lower_snake_case role "
                        "supported by the report; otherwise not_applicable."
                    ),
                },
            ],
            desc=(
                "Determine whether the reporting aircraft performed the specified "
                "maneuver for a qualifying primary trigger, then extract its trigger "
                "and initiating role without inferring facts absent from the report."
            ),
            depends_on=["text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        raw = result_frame(semantic_result)
        raw["qualifies"] = raw["qualifies"].map(parse_bool)
        raw["primary_trigger"] = raw["primary_trigger"].map(
            lambda value: normalize_enum(value, TRIGGERS)
        )
        raw["initiating_role"] = raw["initiating_role"].map(
            normalize_free_label
        )
        extracted = raw.loc[
            raw["qualifies"] & raw["primary_trigger"].notna(),
            ["incident_id", "primary_trigger", "initiating_role"],
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            semantic_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("primary_trigger", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_initiating_role=(
                    "initiating_role",
                    stable_mode,
                ),
            )
            .reset_index()
        )
        trigger_order = {label: index for index, label in enumerate(TRIGGERS)}
        grouped = grouped.sort_values(
            "primary_trigger",
            key=lambda values: values.map(trigger_order),
        ).reset_index(drop=True)
        tracker.record(
            "groupby",
            len(extracted),
            grouped,
        )
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
