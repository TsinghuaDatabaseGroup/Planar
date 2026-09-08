#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-007."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    parse_bool,
    parse_label_list,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-007"
DECISIONS = (
    "attempt_approach",
    "divert",
    "return",
    "hold",
    "leave_holding",
    "continue",
    "declare_minimum_fuel",
    "declare_fuel_emergency",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "text_file",
                "mission",
                "aircraft_operator",
                "result_summary",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[
            (incidents["mission"] == "Passenger")
            & (incidents["aircraft_operator"] == "Air Carrier")
        ].copy()
        tracker.record(
            "FILTER(mission='Passenger' AND aircraft_operator='Air Carrier')",
            len(incidents),
            len(filtered),
            output=filtered,
        )

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(fuel-shaped operational decisions)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["result_summary", "text"],
                output_cols={
                    "qualifies": (
                        "true only when a stated fuel quantity, reserve, burn "
                        "estimate, or remaining-flight-time calculation directly "
                        "shaped at least one listed operational decision"
                    ),
                    "decision_labels": (
                        "a JSON array containing every decision directly shaped by "
                        "the fuel information, chosen only from attempt_approach, "
                        "divert, return, hold, leave_holding, continue, "
                        "declare_minimum_fuel, declare_fuel_emergency; use an empty "
                        "array if none. The structured result summary may support "
                        "diversion, return, or emergency only when the narrative "
                        "confirms the fuel relationship"
                    ),
                    "has_numeric_fuel_value": (
                        "true only if a numeric fuel quantity, reserve, or burn "
                        "value is stated"
                    ),
                    "has_numeric_time_value": (
                        "true only if a numeric remaining-flight-time or other "
                        "fuel-related time value is stated"
                    ),
                },
            )
            raw["qualifies"] = raw["qualifies"].map(parse_bool)
            raw["decision_labels"] = raw["decision_labels"].map(
                lambda value: parse_label_list(value, DECISIONS)
            )
            raw["has_numeric_fuel_value"] = raw[
                "has_numeric_fuel_value"
            ].map(parse_bool)
            raw["has_numeric_time_value"] = raw[
                "has_numeric_time_value"
            ].map(parse_bool)
            qualified = raw.loc[
                raw["qualifies"] & raw["decision_labels"].map(bool),
                [
                    "incident_id",
                    "decision_labels",
                    "has_numeric_fuel_value",
                    "has_numeric_time_value",
                ],
            ]
            extracted = (
                qualified.explode("decision_labels")
                .rename(columns={"decision_labels": "decision_label"})
                .reset_index(drop=True)
            )
            step.set_output(extracted)

        grouped = (
            extracted.groupby("decision_label", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        fuel_counts = (
            extracted[extracted["has_numeric_fuel_value"]]
            .groupby("decision_label")["incident_id"]
            .nunique()
        )
        time_counts = (
            extracted[extracted["has_numeric_time_value"]]
            .groupby("decision_label")["incident_id"]
            .nunique()
        )
        grouped["with_numeric_fuel_value_count"] = (
            grouped["decision_label"].map(fuel_counts).fillna(0).astype(int)
        )
        grouped["with_numeric_time_value_count"] = (
            grouped["decision_label"].map(time_counts).fillna(0).astype(int)
        )
        decision_order = {label: index for index, label in enumerate(DECISIONS)}
        grouped = grouped.sort_values(
            "decision_label",
            key=lambda values: values.map(decision_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([decision_label], COUNT_DISTINCT, COUNT_DISTINCT_IF)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
