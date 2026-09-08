#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-029."""

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
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-029"
GO_AROUND_LABEL = "Flight Crew Executed Go Around / Missed Approach"
NEW_CLEARANCE_LABEL = "Air Traffic Control Issued New Clearance"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "local_time_of_day", "flight_phase"],
        )
        tracker.record("scan", None, incidents)

        afternoon = incidents.loc[
            incidents["local_time_of_day"] == "1201-1800"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), afternoon)

        reports = load_selected_texts("asrs", afternoon)[
            ["incident_id", "flight_phase", "text"]
        ]
        tracker.record("scan", len(afternoon), reports)

        weather_plan = memory_dataset(
            f"{TASK_ID}-weather", reports
        ).sem_filter(
            (
                "Keep this report only if weather or visibility explicitly "
                "caused a go-around, missed approach, or new clearance."
            ),
            depends_on=["text"],
        )
        started = time.time()
        weather_result = weather_plan.run(config)
        weather_candidates = result_frame(weather_result)[
            ["incident_id", "flight_phase", "text"]
        ]
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            weather_candidates,
            weather_result,
            time.time() - started,
        )

        cause_plan = memory_dataset(
            f"{TASK_ID}-cause", weather_candidates
        ).sem_filter(
            (
                "Keep this report only if traffic spacing, runway occupancy, "
                "and aircraft equipment were not the primary cause of the "
                "go-around, missed approach, or new clearance."
            ),
            depends_on=["text"],
        )
        started = time.time()
        cause_result = cause_plan.run(config)
        semantic_candidates = result_frame(cause_result)[
            ["incident_id", "flight_phase", "text"]
        ]
        tracker.record_semantic(
            "sem_filter",
            len(weather_candidates),
            semantic_candidates,
            cause_result,
            time.time() - started,
        )

        environment_attributes = load_table(
            "asrs",
            "environment_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, environment_attributes)

        environment_records = environment_attributes.loc[
            (environment_attributes["attribute"] == "flight_conditions")
            & environment_attributes["value"].isin(["IMC", "Marginal", "Mixed"])
        ].reset_index(drop=True)
        tracker.record(
            "filter", len(environment_attributes), environment_records
        )

        candidates_with_environment = semantic_candidates.merge(
            environment_records[["incident_id", "value"]],
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "join",
            {
                "left": len(semantic_candidates),
                "right": len(environment_records),
            },
            candidates_with_environment,
        )

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        result_events = events.loc[
            (events["event_type"] == "result")
            & events["label"].isin([GO_AROUND_LABEL, NEW_CLEARANCE_LABEL])
        ].reset_index(drop=True)
        tracker.record("filter", len(events), result_events)

        joined = candidates_with_environment.merge(
            result_events[["incident_id", "label"]],
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "join",
            {
                "left": len(candidates_with_environment),
                "right": len(result_events),
            },
            joined,
        )

        approach_or_landing = joined.loc[
            joined["flight_phase"].str.contains(
                "Approach|Landing", case=False, na=False, regex=True
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(joined), approach_or_landing)

        approach_or_landing = approach_or_landing.copy()
        approach_or_landing["recorded_result"] = approach_or_landing[
            "label"
        ].map(
            {
                GO_AROUND_LABEL: "go_around_or_missed_approach",
                NEW_CLEARANCE_LABEL: "new_clearance",
            }
        )
        grouped = (
            approach_or_landing.groupby(
                ["value", "recorded_result"], sort=True
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
            .rename(columns={"value": "flight_conditions"})
        )
        tracker.record("groupby", len(approach_or_landing), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
