#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-029."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-029"
FLIGHT_CONDITIONS = ("IMC", "Marginal", "Mixed")
RESULT_LABELS = {
    "Flight Crew Executed Go Around / Missed Approach": (
        "go_around_or_missed_approach"
    ),
    "Air Traffic Control Issued New Clearance": "new_clearance",
}


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "local_time_of_day", "flight_phase"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        incident_candidates = incidents[
            incidents["local_time_of_day"] == "1201-1800"
        ].copy()
        tracker.record(
            "FILTER(local_time_of_day='1201-1800')",
            len(incidents),
            len(incident_candidates),
            output=incident_candidates,
        )

        reports = load_selected_texts("asrs", incident_candidates)[
            ["incident_id", "flight_phase", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(incident_candidates),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(weather caused target approach or landing result)",
            input_rows=len(reports),
        ) as step:
            weather_caused = reports.sem_filter(
                "The report {text} explicitly states that weather or visibility "
                "caused a go-around, missed approach, or new clearance"
            )
            step.set_output(weather_caused)

        with tracker.step(
            "SEM_FILTER(exclude spacing, occupancy, or equipment primary cause)",
            input_rows=len(weather_caused),
        ) as step:
            semantic_candidates = weather_caused.sem_filter(
                "In the report {text}, traffic spacing, runway occupancy, and "
                "aircraft equipment were not the primary cause of that go-around, "
                "missed approach, or new clearance"
            )
            step.set_output(semantic_candidates)

        environment = load_table("asrs", "environment_attributes.csv")[
            ["incident_id", "attribute", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(environment_attributes.csv)",
            None,
            len(environment),
            output=environment,
        )

        environment_records = environment[
            (environment["attribute"] == "flight_conditions")
            & environment["value"].isin(FLIGHT_CONDITIONS)
        ].copy()
        tracker.record(
            "FILTER(attribute='flight_conditions' AND value IN target conditions)",
            len(environment),
            len(environment_records),
            output=environment_records,
        )

        candidates_with_environment = semantic_candidates.merge(
            environment_records,
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "JOIN(semantic_candidates, environment_records, incident_id)",
            {
                "left": len(semantic_candidates),
                "right": len(environment_records),
            },
            len(candidates_with_environment),
            output=candidates_with_environment,
        )

        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        result_events = events[
            (events["event_type"] == "result")
            & events["label"].isin(RESULT_LABELS)
        ].copy()
        tracker.record(
            "FILTER(event_type='result' AND label IN target results)",
            len(events),
            len(result_events),
            output=result_events,
        )

        joined = candidates_with_environment.merge(
            result_events,
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "JOIN(candidates_with_environment, result_events, incident_id)",
            {
                "left": len(candidates_with_environment),
                "right": len(result_events),
            },
            len(joined),
            output=joined,
        )

        approach_or_landing = joined[
            joined["flight_phase"].str.contains(
                "Approach",
                regex=False,
                na=False,
            )
            | joined["flight_phase"].str.contains(
                "Landing",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(flight_phase CONTAINS Approach OR Landing)",
            len(joined),
            len(approach_or_landing),
            output=approach_or_landing,
        )

        approach_or_landing["flight_conditions"] = approach_or_landing["value"]
        approach_or_landing["recorded_result"] = approach_or_landing[
            "label"
        ].map(RESULT_LABELS)
        grouped = (
            approach_or_landing.groupby(
                ["flight_conditions", "recorded_result"],
                sort=False,
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        condition_order = {
            label: index for index, label in enumerate(FLIGHT_CONDITIONS)
        }
        result_order = {
            "go_around_or_missed_approach": 0,
            "new_clearance": 1,
        }
        grouped = (
            grouped.assign(
                _condition_order=grouped["flight_conditions"].map(
                    condition_order
                ),
                _result_order=grouped["recorded_result"].map(result_order),
            )
            .sort_values(["_condition_order", "_result_order"])
            .drop(columns=["_condition_order", "_result_order"])
            .reset_index(drop=True)
        )
        tracker.record(
            "GROUP_BY([flight_conditions, recorded_result], COUNT_DISTINCT)",
            len(approach_or_landing),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
