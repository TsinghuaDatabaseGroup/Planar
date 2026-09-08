#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-001."""

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
    normalize_free_label,
    parse_bool,
    save_output,
    setup,
    stable_mode,
)

TASK_ID = "aviation_safety-001"
TRIGGERS = (
    "unstable_approach",
    "traffic_spacing",
    "runway_occupancy",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "far_part", "mission"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[
            (incidents["far_part"] == "Part 121")
            & (incidents["mission"] == "Passenger")
        ].copy()
        tracker.record(
            "FILTER(far_part='Part 121' AND mission='Passenger')",
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
            "SEM_EXTRACT(go-around trigger and initiating role)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "qualifies": (
                        "true only when the reporting aircraft actually performed a "
                        "go-around or missed approach whose primary trigger was an "
                        "unstable approach, traffic spacing, or runway occupancy, "
                        "and the primary trigger was not aircraft equipment failure; "
                        "false otherwise"
                    ),
                    "primary_trigger": (
                        "when qualifies is true, exactly one of "
                        "unstable_approach, traffic_spacing, runway_occupancy; "
                        "otherwise not_applicable"
                    ),
                    "initiating_role": (
                        "when qualifies is true, the crew or ATC role that initiated "
                        "the maneuver, as a concise lower_snake_case role supported "
                        "by the report; otherwise not_applicable"
                    ),
                },
            )
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
            step.set_output(extracted)

        grouped = (
            extracted.groupby("primary_trigger", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_initiating_role=("initiating_role", stable_mode),
            )
            .reset_index()
        )
        trigger_order = {label: index for index, label in enumerate(TRIGGERS)}
        grouped = grouped.sort_values(
            "primary_trigger",
            key=lambda values: values.map(trigger_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([primary_trigger], COUNT_DISTINCT, MODE)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
