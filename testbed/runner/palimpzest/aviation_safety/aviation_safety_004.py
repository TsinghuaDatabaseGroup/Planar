#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-004."""

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

TASK_ID = "aviation_safety-004"
FIELDS = ("altitude", "heading", "speed", "runway")
CONSEQUENCES = (
    "altitude_deviation",
    "heading_or_route_deviation",
    "speed_deviation",
    "runway_deviation",
    "other_operational_consequence",
)


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

        filtered = incidents.loc[
            incidents["local_time_of_day"] == "1201-1800"
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
                        "True only when the crew understood one altitude, heading, "
                        "speed, or runway value while ATC, automation, or a procedure "
                        "expected a different value for that same field, and the "
                        "mismatch contributed to a deviation."
                    ),
                },
                {
                    "name": "field_compared",
                    "type": str,
                    "desc": (
                        "When qualifies is true, the primary mismatched field as "
                        "exactly one of altitude, heading, speed, runway; otherwise "
                        "not_applicable."
                    ),
                },
                {
                    "name": "consequence_category",
                    "type": str,
                    "desc": (
                        "When qualifies is true, exactly one resulting consequence "
                        "from altitude_deviation, heading_or_route_deviation, "
                        "speed_deviation, runway_deviation, "
                        "other_operational_consequence; otherwise not_applicable."
                    ),
                },
            ],
            desc=(
                "Identify a mismatch between the crew's understood value and the "
                "value expected by ATC, automation, or procedure for the same flight "
                "field, requiring that this mismatch contributed to a deviation."
            ),
            depends_on=["text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        raw = result_frame(semantic_result)
        raw["qualifies"] = raw["qualifies"].map(parse_bool)
        raw["field_compared"] = raw["field_compared"].map(
            lambda value: normalize_enum(value, FIELDS)
        )
        raw["consequence_category"] = raw["consequence_category"].map(
            lambda value: normalize_enum(value, CONSEQUENCES)
        )
        extracted = raw.loc[
            raw["qualifies"]
            & raw["field_compared"].notna()
            & raw["consequence_category"].notna(),
            ["incident_id", "field_compared", "consequence_category"],
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            semantic_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("field_compared", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_consequence=(
                    "consequence_category",
                    stable_mode,
                ),
            )
            .reset_index()
        )
        field_order = {label: index for index, label in enumerate(FIELDS)}
        grouped = grouped.sort_values(
            "field_compared",
            key=lambda values: values.map(field_order),
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
