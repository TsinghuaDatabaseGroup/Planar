#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-026."""

from __future__ import annotations

import os
import re
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
    stable_mode,
)

TASK_ID = "aviation_safety-026"
MEASUREMENT_TYPES = ("altitude", "speed", "heading", "runway")


def has_numeric_value(value) -> bool:
    return bool(re.search(r"\d", str(value)))


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "state_reference"],
        )
        tracker.record("scan", None, incidents)

        us_reports = incidents.loc[
            incidents["state_reference"] == "US"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), us_reports)

        reports = load_selected_texts("asrs", us_reports)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(us_reports), reports)

        qualifying_plan = memory_dataset(
            f"{TASK_ID}-qualifying", reports
        ).sem_filter(
            (
                "Keep this report only if it contains at least one numeric "
                "cleared-versus-flown or expected-versus-actual comparison for "
                "altitude, speed, heading, or runway, and both values belong to "
                "the same comparison. Nonnumeric or unmatched mentions do not "
                "qualify."
            ),
            depends_on=["text"],
        )
        started = time.time()
        qualifying_result = qualifying_plan.run(config)
        qualifying = result_frame(qualifying_result)[
            ["incident_id", "text"]
        ]
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            qualifying,
            qualifying_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_flat_map(
            cols=[
                {
                    "name": "measurement_type",
                    "type": str,
                    "desc": (
                        "Exactly one of altitude, speed, heading, or runway."
                    ),
                },
                {
                    "name": "expected_value",
                    "type": str,
                    "desc": (
                        "The numeric cleared or expected value, including its "
                        "unit or runway designator when stated."
                    ),
                },
                {
                    "name": "actual_value",
                    "type": str,
                    "desc": (
                        "The numeric flown or actual value from the same "
                        "comparison, including its unit or runway designator "
                        "when stated."
                    ),
                },
                {
                    "name": "deviation",
                    "type": str,
                    "desc": (
                        "A description of the resulting deviation in no more "
                        "than six words."
                    ),
                },
            ],
            desc=(
                "Emit one row for every distinct qualifying comparison in the "
                "report. Keep expected and actual numeric values separate and "
                "do not pair values from different comparisons."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result)
        extracted["measurement_type"] = extracted[
            "measurement_type"
        ].map(lambda value: normalize_enum(value, MEASUREMENT_TYPES))
        for column in ("expected_value", "actual_value", "deviation"):
            extracted[column] = extracted[column].fillna("").astype(str).str.strip()
        tracker.record_semantic(
            "sem_flat_map",
            len(qualifying),
            extracted,
            extraction_result,
            time.time() - started,
        )

        selected = extracted.loc[
            extracted["measurement_type"].notna()
            & extracted["expected_value"].map(has_numeric_value)
            & extracted["actual_value"].map(has_numeric_value)
            & extracted["deviation"].ne(""),
            [
                "incident_id",
                "measurement_type",
                "expected_value",
                "actual_value",
                "deviation",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), selected)

        selected["comparison_key"] = list(
            zip(
                selected["incident_id"],
                selected["expected_value"],
                selected["actual_value"],
            )
        )
        grouped = (
            selected.groupby("measurement_type", sort=False)
            .agg(
                comparison_count=("comparison_key", "nunique"),
                most_common_deviation=("deviation", stable_mode),
            )
            .reset_index()
        )
        type_order = {
            label: index for index, label in enumerate(MEASUREMENT_TYPES)
        }
        grouped = grouped.sort_values(
            "measurement_type",
            key=lambda values: values.map(type_order),
        ).reset_index(drop=True)
        tracker.record("groupby", len(selected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
