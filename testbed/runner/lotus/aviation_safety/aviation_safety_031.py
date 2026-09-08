#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-031."""

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

TASK_ID = "aviation_safety-031"
PHASE_GROUPS = (
    "ground_or_departure",
    "climb_or_cruise",
    "descent_or_approach",
    "landing_or_other",
)
MECHANISMS = (
    "late_or_changed_clearance",
    "misunderstood_clearance",
    "readback_hearback_failure",
    "route_or_heading_confusion",
    "altitude_or_speed_restriction",
)
CLEARANCE_ANOMALIES = {
    "ATC Issue All Types",
    "Deviation / Discrepancy - Procedural Clearance",
}


def phase_group(flight_phase) -> str:
    phase = flight_phase if isinstance(flight_phase, str) else ""
    if any(
        label in phase
        for label in ("Parked", "Taxi", "Takeoff", "Initial Climb")
    ):
        return "ground_or_departure"
    if any(label in phase for label in ("Climb", "Cruise")):
        return "climb_or_cruise"
    if any(label in phase for label in ("Descent", "Approach")):
        return "descent_or_approach"
    return "landing_or_other"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "primary_problem", "flight_phase"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        human_factors = incidents[
            incidents["primary_problem"] == "Human Factors"
        ].copy()
        tracker.record(
            "FILTER(primary_problem='Human Factors')",
            len(incidents),
            len(human_factors),
            output=human_factors,
        )

        reports = load_selected_texts("asrs", human_factors)[
            ["incident_id", "flight_phase", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(human_factors),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(primary clearance-conflict mechanism)",
            input_rows=len(reports),
        ) as step:
            classified = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "clearance_conflict_type": (
                        "assign exactly one clearly primary mechanism from "
                        "late_or_changed_clearance, misunderstood_clearance, "
                        "readback_hearback_failure, route_or_heading_confusion, "
                        "altitude_or_speed_restriction; return not_applicable when "
                        "none is clearly primary"
                    )
                },
            )
            classified["clearance_conflict_type"] = classified[
                "clearance_conflict_type"
            ].map(lambda value: normalize_enum(value, MECHANISMS))
            semantic_mechanisms = classified.loc[
                classified["clearance_conflict_type"].notna(),
                ["incident_id", "flight_phase", "clearance_conflict_type"],
            ].reset_index(drop=True)
            step.set_output(semantic_mechanisms)

        aircraft_attributes = load_table("asrs", "aircraft_attributes.csv")[
            ["incident_id", "attribute", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(aircraft_attributes.csv)",
            None,
            len(aircraft_attributes),
            output=aircraft_attributes,
        )

        ifr_rows = aircraft_attributes[
            (aircraft_attributes["attribute"] == "flight_plan")
            & (aircraft_attributes["value"] == "IFR")
        ].copy()
        tracker.record(
            "FILTER(attribute='flight_plan' AND value='IFR')",
            len(aircraft_attributes),
            len(ifr_rows),
            output=ifr_rows,
        )

        ifr_incidents = ifr_rows[["incident_id"]].drop_duplicates().reset_index(
            drop=True
        )
        tracker.record(
            "DEDUP([incident_id])",
            len(ifr_rows),
            len(ifr_incidents),
            output=ifr_incidents,
        )

        mechanisms_with_ifr = semantic_mechanisms.merge(
            ifr_incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(semantic_mechanisms, ifr_incidents, incident_id)",
            {
                "left": len(semantic_mechanisms),
                "right": len(ifr_incidents),
            },
            len(mechanisms_with_ifr),
            output=mechanisms_with_ifr,
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

        anomaly_rows = events[
            (events["event_type"] == "anomaly")
            & events["label"].isin(CLEARANCE_ANOMALIES)
        ].copy()
        tracker.record(
            "FILTER(event_type='anomaly' AND label IN clearance anomalies)",
            len(events),
            len(anomaly_rows),
            output=anomaly_rows,
        )

        clearance_anomalies = anomaly_rows[
            ["incident_id"]
        ].drop_duplicates().reset_index(drop=True)
        tracker.record(
            "DEDUP([incident_id])",
            len(anomaly_rows),
            len(clearance_anomalies),
            output=clearance_anomalies,
        )

        joined = mechanisms_with_ifr.merge(
            clearance_anomalies,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(mechanisms_with_ifr, clearance_anomalies, incident_id)",
            {
                "left": len(mechanisms_with_ifr),
                "right": len(clearance_anomalies),
            },
            len(joined),
            output=joined,
        )

        projected = joined[
            ["incident_id", "clearance_conflict_type"]
        ].copy()
        projected["flight_phase_group"] = joined["flight_phase"].map(
            phase_group
        )
        tracker.record(
            "PROJECT(clearance_conflict_type, flight_phase_group=CASE)",
            len(joined),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby(
                ["flight_phase_group", "clearance_conflict_type"],
                sort=False,
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        phase_order = {label: index for index, label in enumerate(PHASE_GROUPS)}
        mechanism_order = {
            label: index for index, label in enumerate(MECHANISMS)
        }
        grouped = (
            grouped.assign(
                _phase_order=grouped["flight_phase_group"].map(phase_order),
                _mechanism_order=grouped["clearance_conflict_type"].map(
                    mechanism_order
                ),
            )
            .sort_values(["_phase_order", "_mechanism_order"])
            .drop(columns=["_phase_order", "_mechanism_order"])
            .reset_index(drop=True)
        )
        tracker.record(
            "GROUP_BY([flight_phase_group, clearance_conflict_type], "
            "COUNT_DISTINCT)",
            len(projected),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
