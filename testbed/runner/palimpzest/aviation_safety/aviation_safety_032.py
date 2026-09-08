#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-032."""

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

TASK_ID = "aviation_safety-032"
ISSUE_CATEGORIES = (
    "passenger_medical",
    "security_or_behavior",
    "cabin_equipment_or_door",
    "boarding_or_deplaning",
    "flight_attendant_communication",
)
DECISION_CATEGORIES = (
    "delay_or_cancel",
    "return_or_divert",
    "continue_changed",
    "other_operational_change",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "far_part", "mission"],
        )
        tracker.record("scan", None, incidents)

        part121_passenger = incidents.loc[
            (incidents["far_part"] == "Part 121")
            & (incidents["mission"] == "Passenger")
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), part121_passenger)

        reports = load_selected_texts("asrs", part121_passenger)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(part121_passenger), reports)

        decision_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "qualifies",
                    "type": bool,
                    "desc": (
                        "True only when a cabin-side issue directly caused a "
                        "flight-deck operational change, not merely a cabin-"
                        "service effect."
                    ),
                },
                {
                    "name": "issue_category",
                    "type": str,
                    "desc": (
                        "For a qualifying report, exactly one of "
                        "passenger_medical, security_or_behavior, "
                        "cabin_equipment_or_door, boarding_or_deplaning, or "
                        "flight_attendant_communication."
                    ),
                },
                {
                    "name": "decision_category",
                    "type": str,
                    "desc": (
                        "For a qualifying report, exactly one of delay_or_cancel, "
                        "return_or_divert, continue_changed, or "
                        "other_operational_change."
                    ),
                },
            ],
            desc=(
                "Determine whether a cabin-side issue directly caused a "
                "flight-deck operational change. If so, extract one primary "
                "cabin issue category and one changed-decision category. Exclude "
                "details that affected cabin service only."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        decision_result = decision_plan.run(config)
        extracted = result_frame(decision_result)
        extracted["qualifies"] = extracted["qualifies"].map(parse_bool)
        extracted["issue_category"] = extracted["issue_category"].map(
            lambda value: normalize_enum(value, ISSUE_CATEGORIES)
        )
        extracted["decision_category"] = extracted[
            "decision_category"
        ].map(lambda value: normalize_enum(value, DECISION_CATEGORIES))
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            decision_result,
            time.time() - started,
        )

        semantic_decisions = extracted.loc[
            extracted["qualifies"]
            & extracted["issue_category"].notna()
            & extracted["decision_category"].notna(),
            ["incident_id", "issue_category", "decision_category"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), semantic_decisions)

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        passenger_events = events.loc[
            (events["event_type"] == "passengers_involved")
            & (events["label"] == "Y")
        ].reset_index(drop=True)
        tracker.record("filter", len(events), passenger_events)

        joined = semantic_decisions.merge(
            passenger_events[["incident_id"]],
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "join",
            {
                "left": len(semantic_decisions),
                "right": len(passenger_events),
            },
            joined,
        )

        grouped = (
            joined.groupby(
                ["issue_category", "decision_category"], sort=False
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record("groupby", len(joined), grouped)

        ordered = grouped.sort_values(
            ["incident_count", "issue_category", "decision_category"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        tracker.record("sort", len(grouped), ordered)

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited.copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
