#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-031."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-031"
MECHANISMS = (
    "late_or_changed_clearance",
    "misunderstood_clearance",
    "readback_hearback_failure",
    "route_or_heading_confusion",
    "altitude_or_speed_restriction",
)
CLASSIFICATION_LABELS = (*MECHANISMS, "not_target")


def flight_phase_group(value) -> str:
    phase = "" if pd.isna(value) else str(value)
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


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem", "flight_phase"],
        )
        tracker.record("scan", None, incidents)

        human_factors = incidents.loc[
            incidents["primary_problem"] == "Human Factors"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), human_factors)

        reports = load_selected_texts("asrs", human_factors)[
            ["incident_id", "flight_phase", "text"]
        ]
        tracker.record("scan", len(human_factors), reports)

        mechanism_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "clearance_conflict_type",
                    "type": str,
                    "desc": (
                        "Exactly one clearly primary mechanism from "
                        "late_or_changed_clearance, misunderstood_clearance, "
                        "readback_hearback_failure, route_or_heading_confusion, "
                        "altitude_or_speed_restriction, or not_target when none "
                        "is clearly primary."
                    ),
                }
            ],
            desc=(
                "Classify the single primary clearance-conflict mechanism. "
                "late_or_changed_clearance concerns a clearance issued late or "
                "changed; misunderstood_clearance concerns misunderstanding its "
                "meaning; readback_hearback_failure concerns a failed correction "
                "of an incorrect readback; route_or_heading_confusion concerns "
                "route or heading instructions; altitude_or_speed_restriction "
                "concerns altitude or speed constraints. Use not_target unless "
                "one mechanism is clearly primary."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        mechanism_result = mechanism_plan.run(config)
        classified = result_frame(mechanism_result)
        classified["clearance_conflict_type"] = classified[
            "clearance_conflict_type"
        ].map(lambda value: normalize_enum(value, CLASSIFICATION_LABELS))
        tracker.record_semantic(
            "sem_map",
            len(reports),
            classified,
            mechanism_result,
            time.time() - started,
        )

        semantic_mechanisms = classified.loc[
            classified["clearance_conflict_type"].isin(MECHANISMS),
            ["incident_id", "flight_phase", "clearance_conflict_type"],
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), semantic_mechanisms)

        aircraft_attributes = load_table(
            "asrs",
            "aircraft_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, aircraft_attributes)

        ifr_rows = aircraft_attributes.loc[
            (aircraft_attributes["attribute"] == "flight_plan")
            & (aircraft_attributes["value"] == "IFR")
        ].reset_index(drop=True)
        tracker.record("filter", len(aircraft_attributes), ifr_rows)

        ifr_incidents = ifr_rows[["incident_id"]].drop_duplicates().reset_index(
            drop=True
        )
        tracker.record("distinct", len(ifr_rows), ifr_incidents)

        mechanisms_with_ifr = semantic_mechanisms.merge(
            ifr_incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(semantic_mechanisms),
                "right": len(ifr_incidents),
            },
            mechanisms_with_ifr,
        )

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        anomaly_rows = events.loc[
            (events["event_type"] == "anomaly")
            & events["label"].isin(
                [
                    "ATC Issue All Types",
                    "Deviation / Discrepancy - Procedural Clearance",
                ]
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(events), anomaly_rows)

        clearance_anomalies = anomaly_rows[
            ["incident_id"]
        ].drop_duplicates().reset_index(drop=True)
        tracker.record("distinct", len(anomaly_rows), clearance_anomalies)

        joined = mechanisms_with_ifr.merge(
            clearance_anomalies,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(mechanisms_with_ifr),
                "right": len(clearance_anomalies),
            },
            joined,
        )

        projected = joined[
            ["incident_id", "clearance_conflict_type", "flight_phase"]
        ].copy()
        projected["flight_phase_group"] = projected["flight_phase"].map(
            flight_phase_group
        )
        projected = projected[
            ["incident_id", "clearance_conflict_type", "flight_phase_group"]
        ]
        tracker.record("project", len(joined), projected)

        grouped = (
            projected.groupby(
                ["flight_phase_group", "clearance_conflict_type"], sort=True
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record("groupby", len(projected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
