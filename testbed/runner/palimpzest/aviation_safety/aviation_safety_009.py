#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-009."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-009"
ACTORS = ("atc", "flight_crew", "ground_vehicle_operator")
ACTIONS = (
    "stop_or_hold",
    "reroute_or_turnoff",
    "clearance_or_instruction_correction",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            [
                "incident_id",
                "text_file",
                "aircraft_operator",
                "locale_reference_type",
            ],
        )
        tracker.record("scan", None, incidents)

        filtered = incidents.loc[
            (incidents["aircraft_operator"] == "Air Carrier")
            & (incidents["locale_reference_type"] == "Airport")
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(incidents),
            filtered,
        )

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "scan",
            len(filtered),
            reports,
        )

        semantic = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "qualifies",
                    "type": bool,
                    "desc": (
                        "True only when a taxiway, runway, ramp, or gate surface "
                        "conflict was resolved before the threatened aircraft entered "
                        "the runway, and both the first correcting actor and prevention "
                        "action are clear."
                    ),
                },
                {
                    "name": "correcting_actor",
                    "type": str,
                    "desc": (
                        "When qualifies is true, the first actor to take corrective "
                        "action, exactly one of atc, flight_crew, "
                        "ground_vehicle_operator; otherwise not_applicable."
                    ),
                },
                {
                    "name": "prevention_action",
                    "type": str,
                    "desc": (
                        "When qualifies is true, the action that prevented runway "
                        "entry, exactly one of stop_or_hold, reroute_or_turnoff, "
                        "clearance_or_instruction_correction; otherwise not_applicable."
                    ),
                },
            ],
            desc=(
                "Determine whether the surface conflict was resolved before runway "
                "entry, then identify the first correcting actor and the prevention "
                "action. Do not qualify a report when either classification or the "
                "pre-entry timing is unclear."
            ),
            depends_on=["text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        raw = result_frame(semantic_result)
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
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            semantic_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby(
                ["correcting_actor", "prevention_action"], sort=False
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
            "groupby",
            len(extracted),
            grouped,
        )
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
