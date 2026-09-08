#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-002."""

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

TASK_ID = "aviation_safety-002"
REASONS = (
    "late_clearance",
    "confusing_or_ambiguous",
    "workload_or_timing_pressure",
    "expectation_or_readback_error",
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
                        "True only when the report identifies a specific ATC "
                        "clearance or restriction that could not be followed for a "
                        "non-mechanical reason covered by the permitted reason labels."
                    ),
                },
                {
                    "name": "primary_reason",
                    "type": str,
                    "desc": (
                        "When qualifies is true, exactly one primary reason from "
                        "late_clearance, confusing_or_ambiguous, "
                        "workload_or_timing_pressure, expectation_or_readback_error; "
                        "otherwise not_applicable."
                    ),
                },
                {
                    "name": "has_numeric_time",
                    "type": bool,
                    "desc": (
                        "True only when an explicit numeric amount of available time "
                        "to comply with that clearance or restriction is stated."
                    ),
                },
                {
                    "name": "has_numeric_distance",
                    "type": bool,
                    "desc": (
                        "True only when an explicit numeric amount of available "
                        "distance to comply with that clearance or restriction is stated."
                    ),
                },
            ],
            desc=(
                "Classify the primary non-equipment reason a concrete ATC clearance "
                "or restriction could not be followed, and independently identify "
                "numeric available-time and available-distance evidence. Do not treat "
                "unrelated numbers or mechanical limitations as qualifying evidence."
            ),
            depends_on=["text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        raw = result_frame(semantic_result)
        raw["qualifies"] = raw["qualifies"].map(parse_bool)
        raw["primary_reason"] = raw["primary_reason"].map(
            lambda value: normalize_enum(value, REASONS)
        )
        raw["has_numeric_time"] = raw["has_numeric_time"].map(parse_bool)
        raw["has_numeric_distance"] = raw["has_numeric_distance"].map(
            parse_bool
        )
        extracted = raw.loc[
            raw["qualifies"] & raw["primary_reason"].notna(),
            [
                "incident_id",
                "primary_reason",
                "has_numeric_time",
                "has_numeric_distance",
            ],
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            semantic_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("primary_reason", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                with_numeric_time_count=(
                    "incident_id",
                    lambda values: values[
                        extracted.loc[values.index, "has_numeric_time"]
                    ].nunique(),
                ),
                with_numeric_distance_count=(
                    "incident_id",
                    lambda values: values[
                        extracted.loc[values.index, "has_numeric_distance"]
                    ].nunique(),
                ),
            )
            .reset_index()
        )
        reason_order = {label: index for index, label in enumerate(REASONS)}
        grouped = grouped.sort_values(
            "primary_reason",
            key=lambda values: values.map(reason_order),
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
