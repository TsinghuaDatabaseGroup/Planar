#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-010."""

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
    save_output,
    setup,
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


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "primary_problem"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[
            incidents["primary_problem"] == "Human Factors"
        ].copy()
        tracker.record(
            "FILTER(primary_problem='Human Factors')",
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
            "SEM_EXTRACT(assign cabin-side source by precedence)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "cabin_side_source": (
                        "assign exactly one source only when a cabin-side issue "
                        "affected a flight-deck decision. If several sources apply, "
                        "select the first in this precedence order: medical_event, "
                        "security_issue, policy_issue, door_issue, slide_issue, "
                        "cabin_equipment_issue, passenger_behavior, boarding_issue, "
                        "deplaning_issue, flight_attendant_communication. Return "
                        "not_applicable when no cabin-side issue affected a "
                        "flight-deck decision"
                    )
                },
            )
            raw["cabin_side_source"] = raw["cabin_side_source"].map(
                lambda value: normalize_enum(value, CABIN_SOURCES)
            )
            assigned = raw.loc[
                raw["cabin_side_source"].notna(),
                ["incident_id", "cabin_side_source"],
            ].reset_index(drop=True)
            step.set_output(assigned)

        grouped = (
            assigned.groupby("cabin_side_source", sort=False)
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
            "GROUP_BY([cabin_side_source], COUNT_DISTINCT)",
            len(assigned),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
