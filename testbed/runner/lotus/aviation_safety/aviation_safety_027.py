#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-027."""

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

TASK_ID = "aviation_safety-027"
DECISION_CHANGES = (
    "diversion",
    "airport_change",
    "holding",
    "approach_change",
    "altitude_change",
    "route_change",
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

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(weather primarily caused operational change)",
            input_rows=len(reports),
        ) as step:
            weather_caused = reports.sem_filter(
                "In the report {text}, weather or visibility was the primary cause "
                "of an operational decision change rather than incidental "
                "background"
            )
            step.set_output(weather_caused)

        with tracker.step(
            "SEM_FILTER(exclude passenger-comfort-only turbulence)",
            input_rows=len(weather_caused),
        ) as step:
            operational = weather_caused.sem_filter(
                "The report {text} is not a case where turbulence affected only "
                "passenger comfort without changing the flight's operation"
            )
            step.set_output(operational)

        with tracker.step(
            "SEM_EXTRACT(assign weather-driven decision change)",
            input_rows=len(operational),
        ) as step:
            classified = operational.sem_extract(
                input_cols=["text"],
                output_cols={
                    "decision_change_type": (
                        "assign exactly one change using this precedence: "
                        "diversion, airport_change, holding, approach_change, "
                        "altitude_change, route_change. Return not_applicable if "
                        "none of these operational changes occurred"
                    )
                },
            )
            classified["decision_change_type"] = classified[
                "decision_change_type"
            ].map(lambda value: normalize_enum(value, DECISION_CHANGES))
            assigned = classified.loc[
                classified["decision_change_type"].notna(),
                ["incident_id", "decision_change_type"],
            ].reset_index(drop=True)
            step.set_output(assigned)

        grouped = (
            assigned.groupby("decision_change_type", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        change_order = {
            label: index for index, label in enumerate(DECISION_CHANGES)
        }
        grouped = grouped.sort_values(
            "decision_change_type",
            key=lambda values: values.map(change_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([decision_change_type], COUNT_DISTINCT)",
            len(assigned),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
