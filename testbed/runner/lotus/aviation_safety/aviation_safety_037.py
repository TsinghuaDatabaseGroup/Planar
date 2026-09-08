#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-037."""

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

TASK_ID = "aviation_safety-037"
CANDIDATE_RESULTS = {
    "Air Traffic Control Provided Assistance",
    "Flight Crew Took Evasive Action",
    "Flight Crew Became Reoriented",
    "Flight Crew Overcame Equipment Problem",
    "Air Traffic Control Issued Advisory / Alert",
    "Flight Crew Executed Go Around / Missed Approach",
}
BARRIER_STATUSES = ("successful_barrier", "not_fully_supported")


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        candidate_rows = events[
            (events["event_type"] == "result")
            & events["label"].isin(CANDIDATE_RESULTS)
        ].copy()
        tracker.record(
            "FILTER(event_type='result' AND label IN barrier candidate results)",
            len(events),
            len(candidate_rows),
            output=candidate_rows,
        )

        candidate_events = candidate_rows[
            ["incident_id"]
        ].drop_duplicates().reset_index(drop=True)
        tracker.record(
            "DEDUP([incident_id])",
            len(candidate_rows),
            len(candidate_events),
            output=candidate_events,
        )

        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "flight_phase", "aircraft_operator", "text_file"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        complete_incidents = incidents[
            incidents["flight_phase"].notna()
            & incidents["aircraft_operator"].notna()
        ].copy()
        tracker.record(
            "FILTER(flight_phase IS NOT NULL AND aircraft_operator IS NOT NULL)",
            len(incidents),
            len(complete_incidents),
            output=complete_incidents,
        )

        joined = candidate_events.merge(
            complete_incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(candidate_events, incidents, incident_id)",
            {
                "left": len(candidate_events),
                "right": len(complete_incidents),
            },
            len(joined),
            output=joined,
        )

        reports = load_selected_texts("asrs", joined)[
            ["incident_id", "flight_phase", "aircraft_operator", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=joined.text_file)",
            len(joined),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(successful safety-barrier status)",
            input_rows=len(reports),
        ) as step:
            classified = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "barrier_status": (
                        "classify as successful_barrier only when the narrative "
                        "clearly describes a hazard being detected, an intervention "
                        "being made, and a worse consequence being avoided or the "
                        "situation being stabilized; otherwise classify as "
                        "not_fully_supported"
                    )
                },
            )
            classified["barrier_status"] = classified["barrier_status"].map(
                lambda value: normalize_enum(value, BARRIER_STATUSES)
            )
            classified["barrier_status"] = classified[
                "barrier_status"
            ].fillna("not_fully_supported")
            classified = classified[
                [
                    "incident_id",
                    "flight_phase",
                    "aircraft_operator",
                    "barrier_status",
                ]
            ].reset_index(drop=True)
            step.set_output(classified)

        classified["_successful_incident"] = classified["incident_id"].where(
            classified["barrier_status"] == "successful_barrier"
        )
        grouped = (
            classified.groupby(
                ["flight_phase", "aircraft_operator"],
                sort=False,
            )
            .agg(
                candidate_incident_count=("incident_id", "nunique"),
                successful_barrier_count=("_successful_incident", "nunique"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([flight_phase, aircraft_operator], candidate/success counts)",
            len(classified),
            len(grouped),
            output=grouped,
        )

        eligible = grouped[grouped["candidate_incident_count"] >= 20].copy()
        tracker.record(
            "FILTER(candidate_incident_count >= 20)",
            len(grouped),
            len(eligible),
            output=eligible,
        )

        projected = eligible.copy()
        projected["successful_barrier_rate"] = (
            projected["successful_barrier_count"]
            / projected["candidate_incident_count"]
        )
        tracker.record(
            "PROJECT(successful_barrier_rate=successful/candidate)",
            len(eligible),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            [
                "successful_barrier_rate",
                "successful_barrier_count",
                "flight_phase",
                "aircraft_operator",
            ],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([rate DESC, success count DESC, phase/operator ASC])",
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

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and safety-barrier summary fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
