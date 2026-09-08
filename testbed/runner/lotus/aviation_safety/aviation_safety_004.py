#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-004."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    parse_bool,
    save_output,
    setup,
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


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "local_time_of_day"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[
            incidents["local_time_of_day"] == "1201-1800"
        ].copy()
        tracker.record(
            "FILTER(local_time_of_day='1201-1800')",
            len(incidents),
            len(filtered),
            output=filtered,
        )

        reports = load_selected_texts("asrs", filtered)[["incident_id", "text"]]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(mismatch field and deviation consequence)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "qualifies": (
                        "true only when the crew understood one value for an "
                        "altitude, heading, speed, or runway while ATC, automation, "
                        "or a procedure expected a different value for that same "
                        "field, and the mismatch contributed to a deviation"
                    ),
                    "field_compared": (
                        "when qualifies is true, exactly one of altitude, heading, "
                        "speed, runway; if several are mentioned, choose the field "
                        "whose mismatch most directly contributed to the deviation; "
                        "otherwise not_applicable"
                    ),
                    "consequence_category": (
                        "when qualifies is true, exactly one of altitude_deviation, "
                        "heading_or_route_deviation, speed_deviation, "
                        "runway_deviation, other_operational_consequence; otherwise "
                        "not_applicable"
                    ),
                },
            )
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
            step.set_output(extracted)

        grouped = (
            extracted.groupby("field_compared", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_consequence=("consequence_category", stable_mode),
            )
            .reset_index()
        )
        field_order = {label: index for index, label in enumerate(FIELDS)}
        grouped = grouped.sort_values(
            "field_compared",
            key=lambda values: values.map(field_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([field_compared], COUNT_DISTINCT, MODE)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
