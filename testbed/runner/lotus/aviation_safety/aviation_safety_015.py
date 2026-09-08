#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-015."""

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
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-015"
CONSEQUENCES = (
    "emergency_declaration",
    "diversion",
    "return",
    "holding",
    "continued_with_concern",
)


def numeric_values(value) -> list[str]:
    return [
        item
        for item in parse_string_list(value)
        if any(character.isdigit() for character in item)
    ]


def main():
    setup(max_tokens=4096)
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
            "SEM_EXTRACT(fuel values, time values, and consequence)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["result_summary", "text"],
                output_cols={
                    "qualifies": (
                        "true only when a fuel quantity, reserve, calculation, or "
                        "dispatch-release decision directly affected the operation"
                    ),
                    "fuel_values_normalized": (
                        "a JSON array of the explicitly stated numeric fuel values, "
                        "each as a short normalized string with its unit; use an "
                        "empty array when no numeric fuel value is stated"
                    ),
                    "time_values_normalized": (
                        "a JSON array of the explicitly stated numeric fuel-related "
                        "time values, each as a short normalized string with its "
                        "unit; use an empty array when no numeric time value is stated"
                    ),
                    "operational_consequence": (
                        "when qualifies is true, choose the highest applicable "
                        "consequence using this priority: emergency_declaration, "
                        "diversion, return, holding, continued_with_concern; otherwise "
                        "not_applicable. Use structured emergency, diversion, or "
                        "return results only when consistent with the narrative"
                    ),
                },
            )
            raw["qualifies"] = raw["qualifies"].map(parse_bool)
            raw["fuel_values_normalized"] = raw[
                "fuel_values_normalized"
            ].map(numeric_values)
            raw["time_values_normalized"] = raw[
                "time_values_normalized"
            ].map(numeric_values)
            raw["operational_consequence"] = raw[
                "operational_consequence"
            ].map(lambda value: normalize_enum(value, CONSEQUENCES))
            extracted = raw.loc[
                raw["qualifies"] & raw["operational_consequence"].notna(),
                [
                    "incident_id",
                    "fuel_values_normalized",
                    "time_values_normalized",
                    "operational_consequence",
                ],
            ].reset_index(drop=True)
            step.set_output(extracted)

        consequence_priority = {
            label: index + 1 for index, label in enumerate(CONSEQUENCES)
        }
        projected = extracted.copy()
        projected["consequence_priority"] = projected[
            "operational_consequence"
        ].map(consequence_priority)
        projected["has_numeric_value"] = projected[
            "fuel_values_normalized"
        ].map(bool) | projected["time_values_normalized"].map(bool)
        tracker.record(
            "PROJECT(consequence_priority, has_numeric_value)",
            len(extracted),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            ["consequence_priority", "has_numeric_value", "incident_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([consequence_priority ASC, has_numeric_value DESC, "
            "incident_id ASC])",
            len(projected),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(5).copy()
        tracker.record(
            "LIMIT(5)",
            len(ordered),
            len(limited),
            output=limited,
        )

        result = limited[
            [
                "incident_id",
                "fuel_values_normalized",
                "time_values_normalized",
                "operational_consequence",
            ]
        ].copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank, incident_id, fuel_values, time_values, consequence)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
