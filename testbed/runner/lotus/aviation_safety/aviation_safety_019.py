#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-019."""

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
)


TASK_ID = "aviation_safety-019"
CONSEQUENCES = (
    "emergency_declaration",
    "diversion",
    "return",
    "holding",
    "continued_with_concern",
)


def main():
    setup(max_tokens=512)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "text_file",
                "far_part",
                "mission",
                "result_summary",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        passenger_reports = incidents[
            (incidents["far_part"] == "Part 121")
            & (incidents["mission"] == "Passenger")
        ].copy()
        tracker.record(
            "FILTER(far_part='Part 121' AND mission='Passenger')",
            len(incidents),
            len(passenger_reports),
            output=passenger_reports,
        )

        reports = load_selected_texts("asrs", passenger_reports)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(passenger_reports),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(primary fuel-related operational consequence)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["result_summary", "text"],
                output_cols={
                    "operational_consequence": (
                        "When a fuel quantity, reserve, calculation, or "
                        "dispatch-release decision directly affected the "
                        "operation, output the highest applicable consequence "
                        "using this strict precedence: emergency_declaration, "
                        "then diversion, then return, then holding, then "
                        "continued_with_concern. Otherwise output not_applicable."
                    ),
                    "has_numeric_fuel_value": (
                        "true only when the qualifying report states a numeric "
                        "fuel value; false otherwise"
                    ),
                    "has_numeric_time_value": (
                        "true only when the qualifying report states a numeric "
                        "time value; false otherwise"
                    ),
                },
            )
            raw["operational_consequence"] = raw[
                "operational_consequence"
            ].map(lambda value: normalize_enum(value, CONSEQUENCES))
            raw["has_numeric_fuel_value"] = raw[
                "has_numeric_fuel_value"
            ].map(parse_bool)
            raw["has_numeric_time_value"] = raw[
                "has_numeric_time_value"
            ].map(parse_bool)
            extracted = raw.loc[
                raw["operational_consequence"].notna(),
                [
                    "incident_id",
                    "operational_consequence",
                    "has_numeric_fuel_value",
                    "has_numeric_time_value",
                ],
            ].reset_index(drop=True)
            step.set_output(extracted)

        grouped = (
            extracted.groupby("operational_consequence", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                with_numeric_fuel_value_count=(
                    "incident_id",
                    lambda ids: ids[
                        extracted.loc[ids.index, "has_numeric_fuel_value"]
                    ].nunique(),
                ),
                with_numeric_time_value_count=(
                    "incident_id",
                    lambda ids: ids[
                        extracted.loc[ids.index, "has_numeric_time_value"]
                    ].nunique(),
                ),
            )
            .reset_index()
        )
        consequence_order = {
            label: index for index, label in enumerate(CONSEQUENCES)
        }
        grouped = grouped.sort_values(
            "operational_consequence",
            key=lambda values: values.map(consequence_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY(operational_consequence, distinct incident counts)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
