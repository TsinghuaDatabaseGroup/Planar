#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-041."""

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

TASK_ID = "aviation_safety-041"
CLEARANCE_EVENT_LABELS = (
    "ATC Issue All Types",
    "Deviation / Discrepancy - Procedural Clearance",
)
CONFLICT_TYPES = (
    "late_or_changed_clearance",
    "ambiguous_or_incomplete_clearance",
    "misunderstood_clearance",
    "readback_hearback_failure",
)


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
        aircraft_attributes = load_table(
            "asrs",
            "aircraft_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, aircraft_attributes)

        ifr_rows = aircraft_attributes.loc[
            (aircraft_attributes["attribute"] == "flight_plan")
            & (aircraft_attributes["value"] == "IFR"),
            ["incident_id"],
        ].reset_index(drop=True)
        tracker.record("filter", len(aircraft_attributes), ifr_rows)

        ifr_flights = ifr_rows.drop_duplicates(
            subset=["incident_id"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(ifr_rows), ifr_flights)

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        clearance_rows = events.loc[
            (events["event_type"] == "anomaly")
            & events["label"].isin(CLEARANCE_EVENT_LABELS)
        ].reset_index(drop=True)
        tracker.record("filter", len(events), clearance_rows)

        clearance_events = (
            clearance_rows.groupby("incident_id", sort=False)
            .agg(clearance_event_labels=("label", list))
            .reset_index()
        )
        tracker.record("groupby", len(clearance_rows), clearance_events)

        ifr_clearance_candidates = ifr_flights.merge(
            clearance_events,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {"left": len(ifr_flights), "right": len(clearance_events)},
            ifr_clearance_candidates,
        )

        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "flight_phase"],
        )
        tracker.record("scan", None, incidents)

        candidate_reports = ifr_clearance_candidates.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(ifr_clearance_candidates),
                "right": len(incidents),
            },
            candidate_reports,
        )

        candidate_narratives = load_selected_texts("asrs", candidate_reports)[
            ["incident_id", "flight_phase", "clearance_event_labels", "text"]
        ]
        tracker.record("scan", len(candidate_reports), candidate_narratives)

        problematic_plan = memory_dataset(
            f"{TASK_ID}-problematic-clearance", candidate_narratives
        ).sem_filter(
            filter=(
                "The report describes an ATC clearance or instruction that was "
                "late, confusing, ambiguous, incomplete, misunderstood, or "
                "involved an uncorrected readback/hearback failure."
            ),
            depends_on=["clearance_event_labels", "text"],
        )
        started = time.time()
        problematic_result = problematic_plan.run(config)
        problematic_clearances = result_frame(
            problematic_result,
            candidate_narratives,
        )
        tracker.record_semantic(
            "sem_filter",
            len(candidate_narratives),
            problematic_clearances,
            problematic_result,
            time.time() - started,
        )

        deviation_plan = memory_dataset(
            f"{TASK_ID}-actual-deviation", problematic_clearances
        ).sem_filter(
            filter=(
                "The clearance problem subsequently triggered an actual deviation "
                "from an assigned altitude, heading or track, or airspace boundary. "
                "Exclude potential deviations corrected before occurrence and cases "
                "where ATC only detected or corrected an independently caused "
                "deviation."
            ),
            depends_on=["text"],
        )
        started = time.time()
        deviation_result = deviation_plan.run(config)
        actual_deviations = result_frame(
            deviation_result,
            problematic_clearances,
        )
        tracker.record_semantic(
            "sem_filter",
            len(problematic_clearances),
            actual_deviations,
            deviation_result,
            time.time() - started,
        )

        conflict_plan = memory_dataset(
            f"{TASK_ID}-conflict-type", actual_deviations
        ).sem_map(
            cols=[
                {
                    "name": "clearance_conflict_type",
                    "type": str,
                    "desc": (
                        "Exactly one of late_or_changed_clearance, "
                        "ambiguous_or_incomplete_clearance, "
                        "misunderstood_clearance, or readback_hearback_failure."
                    ),
                },
                {
                    "name": "evidence",
                    "type": str,
                    "desc": "Concise narrative evidence supporting the classification.",
                },
            ],
            desc=(
                "Classify the clearance conflict that triggered the actual "
                "deviation and extract concise supporting evidence."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        conflict_result = conflict_plan.run(config)
        typed_conflicts = result_frame(
            conflict_result,
            actual_deviations,
            ["clearance_conflict_type", "evidence"],
        )
        typed_conflicts["clearance_conflict_type"] = typed_conflicts[
            "clearance_conflict_type"
        ].map(lambda value: normalize_enum(value, CONFLICT_TYPES))
        tracker.record_semantic(
            "sem_map",
            len(actual_deviations),
            typed_conflicts,
            conflict_result,
            time.time() - started,
        )

        selected_conflicts = typed_conflicts.loc[
            typed_conflicts["clearance_conflict_type"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(typed_conflicts), selected_conflicts)

        selected_conflicts = selected_conflicts.copy()
        selected_conflicts["flight_phase"] = selected_conflicts[
            "flight_phase"
        ].map(flight_phase_group)
        grouped = (
            selected_conflicts.groupby(
                ["flight_phase", "clearance_conflict_type"], sort=True
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record("groupby", len(selected_conflicts), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
