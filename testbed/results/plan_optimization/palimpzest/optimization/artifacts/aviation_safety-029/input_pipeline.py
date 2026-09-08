#!/usr/bin/env python3
"""Plan-optimization pipeline for aviation_safety-029."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import palimpzest as pz  # noqa: E402
from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    run_plan_optimization,
    save_output,
)

TASK_ID = "aviation_safety-029"
GO_AROUND_LABEL = "Flight Crew Executed Go Around / Missed Approach"
NEW_CLEARANCE_LABEL = "Air Traffic Control Issued New Clearance"


def is_approach_or_landing(record: dict) -> bool:
    value = str(record.get("flight_phase") or "").casefold()
    return "approach" in value or "landing" in value


def add_recorded_result(record: dict) -> dict:
    return {
        "recorded_result": (
            "go_around_or_missed_approach"
            if record.get("label") == GO_AROUND_LABEL
            else "new_clearance"
        )
    }


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "local_time_of_day", "flight_phase"],
        )
        afternoon = incidents.loc[
            incidents["local_time_of_day"] == "1201-1800"
        ].reset_index(drop=True)
        reports = load_selected_texts("asrs", afternoon)[
            ["incident_id", "flight_phase", "text"]
        ]

        environment = load_table(
            "asrs",
            "environment_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        environment = environment.loc[
            (environment["attribute"] == "flight_conditions")
            & environment["value"].isin(["IMC", "Marginal", "Mixed"]),
            ["incident_id", "value"],
        ].reset_index(drop=True)

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        result_events = events.loc[
            (events["event_type"] == "result")
            & events["label"].isin([GO_AROUND_LABEL, NEW_CLEARANCE_LABEL]),
            ["incident_id", "label"],
        ].reset_index(drop=True)

        candidates = memory_dataset(f"{TASK_ID}-reports", reports).sem_filter(
            (
                "Keep this report only if weather or visibility explicitly "
                "caused a go-around, missed approach, or new clearance."
            ),
            depends_on=["text"],
        )
        candidates = candidates.sem_filter(
            (
                "Keep this report only if traffic spacing, runway occupancy, "
                "and aircraft equipment were not the primary cause of that "
                "result."
            ),
            depends_on=["text"],
        )
        candidates = candidates.join(
            memory_dataset(f"{TASK_ID}-environment", environment),
            on="incident_id",
            how="inner",
        )
        joined = candidates.join(
            memory_dataset(f"{TASK_ID}-events", result_events),
            on="incident_id",
            how="inner",
        )
        joined = joined.filter(
            is_approach_or_landing,
            depends_on=["flight_phase"],
        )
        joined = joined.map(
            add_recorded_result,
            cols=[
                {
                    "name": "recorded_result",
                    "type": str,
                    "desc": "Normalized recorded result label.",
                }
            ],
            depends_on=["label"],
        )
        joined = joined.distinct(["incident_id", "value", "recorded_result"])
        plan = joined.groupby(
            pz.GroupBySig(
                group_by_fields=["value", "recorded_result"],
                agg_funcs=["count"],
                agg_fields=["incident_id"],
            )
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True).rename(
            columns={
                "value": "flight_conditions",
                "count(incident_id)": "incident_count",
            }
        )
        tracker.record_semantic(
            "optimized_plan",
            {
                "reports": len(reports),
                "environment": len(environment),
                "events": len(result_events),
            },
            output,
            optimized.result,
            time.time() - started,
        )
        answer = df_records(
            output[["flight_conditions", "recorded_result", "incident_count"]]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
