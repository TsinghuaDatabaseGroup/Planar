#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-041."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-041"
CLEARANCE_EVENT_LABELS = {
    "ATC Issue All Types",
    "Deviation / Discrepancy - Procedural Clearance",
}
CONFLICT_TYPES = (
    "late_or_changed_clearance",
    "ambiguous_or_incomplete_clearance",
    "misunderstood_clearance",
    "readback_hearback_failure",
)


def phase_group(value) -> str:
    phase = str(value)
    if any(token in phase for token in ("Parked", "Taxi", "Takeoff", "Initial Climb")):
        return "ground_or_departure"
    if any(token in phase for token in ("Climb", "Cruise")):
        return "climb_or_cruise"
    if any(token in phase for token in ("Descent", "Approach")):
        return "descent_or_approach"
    return "landing_or_other"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
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

        ifr_flights = ifr_rows[["incident_id"]].drop_duplicates().reset_index(
            drop=True
        )
        tracker.record(
            "DEDUP([incident_id])",
            len(ifr_rows),
            len(ifr_flights),
            output=ifr_flights,
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

        clearance_rows = events[
            (events["event_type"] == "anomaly")
            & events["label"].isin(CLEARANCE_EVENT_LABELS)
        ].copy()
        tracker.record(
            "FILTER(event_type='anomaly' AND label IN clearance labels)",
            len(events),
            len(clearance_rows),
            output=clearance_rows,
        )

        clearance_events = (
            clearance_rows.groupby("incident_id", sort=False)
            .agg(
                clearance_event_labels=(
                    "label",
                    lambda values: list(dict.fromkeys(values.dropna())),
                )
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(label))",
            len(clearance_rows),
            len(clearance_events),
            output=clearance_events,
        )

        ifr_clearance_candidates = ifr_flights.merge(
            clearance_events,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(ifr_flights, clearance_events, incident_id)",
            {"left": len(ifr_flights), "right": len(clearance_events)},
            len(ifr_clearance_candidates),
            output=ifr_clearance_candidates,
        )

        incident_records = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "flight_phase"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incident_records),
            output=incident_records,
        )

        candidate_reports = ifr_clearance_candidates.merge(
            incident_records,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(ifr_clearance_candidates, incident_records, incident_id)",
            {
                "left": len(ifr_clearance_candidates),
                "right": len(incident_records),
            },
            len(candidate_reports),
            output=candidate_reports,
        )

        candidate_narratives = load_selected_texts("asrs", candidate_reports)[
            ["incident_id", "clearance_event_labels", "flight_phase", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=candidate_reports.text_file)",
            len(candidate_reports),
            len(candidate_narratives),
            output=candidate_narratives,
        )

        with tracker.step(
            "SEM_FILTER(late, confusing, or ambiguous ATC clearance)",
            input_rows=len(candidate_narratives),
        ) as step:
            if candidate_narratives.empty:
                problematic_clearances = candidate_narratives.copy()
            else:
                problematic_clearances = candidate_narratives.sem_filter(
                    "Keep the report with recorded clearance events "
                    "{clearance_event_labels} only when narrative {text} describes "
                    "an ATC clearance or instruction that was late, confusing, "
                    "ambiguous, incomplete, misunderstood, or involved an "
                    "uncorrected readback/hearback failure."
                )
            step.set_output(problematic_clearances)

        with tracker.step(
            "SEM_FILTER(clearance problem caused actual deviation)",
            input_rows=len(problematic_clearances),
        ) as step:
            if problematic_clearances.empty:
                actual_deviations = problematic_clearances.copy()
            else:
                actual_deviations = problematic_clearances.sem_filter(
                    "Keep report {text} only when that clearance problem "
                    "subsequently caused an actual deviation from an assigned "
                    "altitude, heading or track, or airspace boundary. Exclude a "
                    "potential deviation corrected before it occurred and a case "
                    "where ATC only detected or corrected a deviation caused "
                    "independently of the clearance problem."
                )
            step.set_output(actual_deviations)

        with tracker.step(
            "SEM_EXTRACT(clearance conflict type and evidence)",
            input_rows=len(actual_deviations),
        ) as step:
            if actual_deviations.empty:
                typed_conflicts = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "flight_phase",
                        "clearance_conflict_type",
                        "evidence",
                    ]
                )
            else:
                extracted = actual_deviations.sem_extract(
                    input_cols=["text"],
                    output_cols={
                        "clearance_conflict_type": (
                            "exactly one primary conflict type: "
                            "late_or_changed_clearance, "
                            "ambiguous_or_incomplete_clearance, "
                            "misunderstood_clearance, or "
                            "readback_hearback_failure"
                        ),
                        "evidence": (
                            "concise narrative evidence that identifies the "
                            "clearance problem and the resulting actual deviation"
                        ),
                    },
                )
                extracted["clearance_conflict_type"] = extracted[
                    "clearance_conflict_type"
                ].map(lambda value: normalize_enum(value, CONFLICT_TYPES))
                extracted["evidence"] = extracted["evidence"].map(clean_text)
                typed_conflicts = extracted.loc[
                    extracted["clearance_conflict_type"].notna(),
                    [
                        "incident_id",
                        "flight_phase",
                        "clearance_conflict_type",
                        "evidence",
                    ],
                ].reset_index(drop=True)
            step.set_output(typed_conflicts)

        projected = typed_conflicts.copy()
        projected["flight_phase"] = projected["flight_phase"].map(phase_group)
        tracker.record(
            "PROJECT(flight_phase=CASE phase labels)",
            len(typed_conflicts),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby(
                ["flight_phase", "clearance_conflict_type"],
                sort=False,
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([flight_phase, clearance_conflict_type], COUNT_DISTINCT)",
            len(projected),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
