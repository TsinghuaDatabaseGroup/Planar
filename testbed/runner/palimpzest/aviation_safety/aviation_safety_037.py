#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-037."""

from __future__ import annotations

import os
import sys
import time

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

TASK_ID = "aviation_safety-037"

RESULT_LABELS = {
    "Air Traffic Control Provided Assistance",
    "Flight Crew Took Evasive Action",
    "Flight Crew Became Reoriented",
    "Flight Crew Overcame Equipment Problem",
    "Air Traffic Control Issued Advisory / Alert",
    "Flight Crew Executed Go Around / Missed Approach",
}

BARRIER_LABELS = {
    "successful_barrier",
    "not_fully_supported",
}


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        result_events = events.loc[
            events["event_type"].eq("result") & events["label"].isin(RESULT_LABELS),
            ["incident_id"],
        ].reset_index(drop=True)
        tracker.record("filter", len(events), result_events)

        candidate_events = result_events.drop_duplicates(
            subset=["incident_id"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(result_events), candidate_events)

        incidents = load_table(
            "asrs",
            "incidents.csv",
            [
                "incident_id",
                "flight_phase",
                "aircraft_operator",
                "text_file",
            ],
        )
        tracker.record("scan", None, incidents)

        selected_incidents = incidents.loc[
            incidents["flight_phase"].notna()
            & incidents["aircraft_operator"].notna(),
            ["incident_id", "flight_phase", "aircraft_operator", "text_file"],
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), selected_incidents)

        candidates = candidate_events.merge(
            selected_incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(candidate_events),
                "right": len(selected_incidents),
            },
            candidates,
        )

        documents = load_selected_texts("asrs", candidates)[
            ["incident_id", "flight_phase", "aircraft_operator", "text"]
        ]
        tracker.record("scan", len(candidates), documents)

        classification_plan = memory_dataset(TASK_ID, documents).sem_map(
            cols=[
                {
                    "name": "barrier_status",
                    "type": str,
                    "desc": (
                        "Exactly one of successful_barrier or "
                        "not_fully_supported."
                    ),
                }
            ],
            desc=(
                "Classify the candidate as successful_barrier only when its "
                "narrative clearly describes all three required elements: a "
                "hazard was detected, an intervention was made, and a worse "
                "consequence was avoided or the situation was stabilized. "
                "Otherwise classify it as not_fully_supported."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(classification_result)
        classified["barrier_status"] = classified["barrier_status"].map(
            lambda value: normalize_enum(value, BARRIER_LABELS)
        )
        tracker.record_semantic(
            "sem_map",
            len(documents),
            classified,
            classification_result,
            time.time() - started,
        )

        valid_classifications = classified.loc[
            classified["barrier_status"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), valid_classifications)

        valid_classifications = valid_classifications.copy()
        valid_classifications["successful_incident_id"] = valid_classifications[
            "incident_id"
        ].where(
            valid_classifications["barrier_status"].eq("successful_barrier")
        )
        grouped = (
            valid_classifications.groupby(
                ["flight_phase", "aircraft_operator"], sort=False
            )
            .agg(
                candidate_incident_count=("incident_id", "nunique"),
                successful_barrier_count=("successful_incident_id", "nunique"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(valid_classifications), grouped)

        eligible = grouped.loc[
            grouped["candidate_incident_count"].ge(20)
        ].reset_index(drop=True)
        tracker.record("filter", len(grouped), eligible)

        eligible["successful_safety_barrier_rate"] = (
            eligible["successful_barrier_count"]
            / eligible["candidate_incident_count"]
        )
        projected = eligible[
            [
                "flight_phase",
                "aircraft_operator",
                "candidate_incident_count",
                "successful_barrier_count",
                "successful_safety_barrier_rate",
            ]
        ].copy()
        tracker.record("project", len(eligible), projected)

        ordered = projected.sort_values(
            by=[
                "successful_safety_barrier_rate",
                "successful_barrier_count",
                "flight_phase",
                "aircraft_operator",
            ],
            ascending=[False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("sort", len(projected), ordered)

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_five)

        answer_frame = top_five.copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(top_five), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
