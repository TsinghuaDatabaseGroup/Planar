#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-009."""

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

TASK_ID = "aviation_safety-009"
ACTORS = ("atc", "flight_crew", "ground_vehicle_operator")
ACTIONS = (
    "stop_or_hold",
    "reroute_or_turnoff",
    "clearance_or_instruction_correction",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "text_file",
                "aircraft_operator",
                "locale_reference_type",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[
            (incidents["aircraft_operator"] == "Air Carrier")
            & (incidents["locale_reference_type"] == "Airport")
        ].copy()
        tracker.record(
            "FILTER(aircraft_operator='Air Carrier' AND "
            "locale_reference_type='Airport')",
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
            "SEM_EXTRACT(surface-conflict correcting actor and action)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "qualifies": (
                        "true only when a taxiway, runway, ramp, or gate conflict "
                        "was resolved before the threatened aircraft entered the "
                        "runway, and both the first correcting actor and prevention "
                        "action are clear"
                    ),
                    "correcting_actor": (
                        "when qualifies is true, the first actor who corrected the "
                        "conflict, exactly one of atc, flight_crew, "
                        "ground_vehicle_operator; otherwise not_applicable"
                    ),
                    "prevention_action": (
                        "when qualifies is true, exactly one of stop_or_hold, "
                        "reroute_or_turnoff, clearance_or_instruction_correction; "
                        "otherwise not_applicable"
                    ),
                },
            )
            raw["qualifies"] = raw["qualifies"].map(parse_bool)
            raw["correcting_actor"] = raw["correcting_actor"].map(
                lambda value: normalize_enum(value, ACTORS)
            )
            raw["prevention_action"] = raw["prevention_action"].map(
                lambda value: normalize_enum(value, ACTIONS)
            )
            extracted = raw.loc[
                raw["qualifies"]
                & raw["correcting_actor"].notna()
                & raw["prevention_action"].notna(),
                ["incident_id", "correcting_actor", "prevention_action"],
            ].reset_index(drop=True)
            step.set_output(extracted)

        grouped = (
            extracted.groupby(
                ["correcting_actor", "prevention_action"],
                sort=False,
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        actor_order = {label: index for index, label in enumerate(ACTORS)}
        action_order = {label: index for index, label in enumerate(ACTIONS)}
        grouped = (
            grouped.assign(
                _actor_order=grouped["correcting_actor"].map(actor_order),
                _action_order=grouped["prevention_action"].map(action_order),
            )
            .sort_values(["_actor_order", "_action_order"])
            .drop(columns=["_actor_order", "_action_order"])
            .reset_index(drop=True)
        )
        tracker.record(
            "GROUP_BY([correcting_actor, prevention_action], COUNT_DISTINCT)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
