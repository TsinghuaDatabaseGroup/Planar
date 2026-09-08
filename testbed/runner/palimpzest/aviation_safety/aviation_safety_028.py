#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-028."""

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

TASK_ID = "aviation_safety-028"
RESPONSES = (
    "followed_automation",
    "overrode_or_disconnected",
    "sought_atc_clarification",
)


def structured_response(event_pairs) -> str | None:
    labels = {label for _event_type, label in event_pairs}
    if "Flight Crew FLC complied w / Automation / Advisory" in labels:
        return "followed_automation"
    if labels.intersection(
        {"Flight Crew FLC Overrode Automation", "Flight Crew Overrode Automation"}
    ):
        return "overrode_or_disconnected"
    return None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "far_part", "locale_reference_type"],
        )
        tracker.record("scan", None, incidents)

        airport_part121 = incidents.loc[
            (incidents["far_part"] == "Part 121")
            & (incidents["locale_reference_type"] == "Airport")
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), airport_part121)

        reports = load_selected_texts("asrs", airport_part121)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(airport_part121), reports)

        response_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "narrative_response",
                    "type": str,
                    "desc": (
                        "When a crew action is clear, exactly one of "
                        "followed_automation, overrode_or_disconnected, or "
                        "sought_atc_clarification, using that precedence. Use "
                        "none when no such narrative action is clear."
                    ),
                }
            ],
            desc=(
                "Extract the single dominant crew response to automation from "
                "the narrative. Do not infer a response when the action is not "
                "clear."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        response_result = response_plan.run(config)
        semantic_responses = result_frame(response_result)
        semantic_responses["narrative_response"] = semantic_responses[
            "narrative_response"
        ].map(lambda value: normalize_enum(value, RESPONSES))
        semantic_responses = semantic_responses[
            ["incident_id", "narrative_response"]
        ]
        tracker.record_semantic(
            "sem_map",
            len(reports),
            semantic_responses,
            response_result,
            time.time() - started,
        )

        aircraft_attributes = load_table(
            "asrs",
            "aircraft_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, aircraft_attributes)

        navigation_rows = aircraft_attributes.loc[
            (aircraft_attributes["attribute"] == "nav_in_use")
            & (
                aircraft_attributes["value"].str.contains(
                    "FMS Or FMC", na=False
                )
                | aircraft_attributes["value"].str.contains("GPS", na=False)
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(aircraft_attributes), navigation_rows)

        navigation_records = (
            navigation_rows.groupby("incident_id", sort=False)
            .agg(
                navigation_in_use=(
                    "value",
                    lambda values: list(dict.fromkeys(values)),
                )
            )
            .reset_index()
        )
        tracker.record("groupby", len(navigation_rows), navigation_records)

        responses_with_navigation = semantic_responses.merge(
            navigation_records,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(semantic_responses),
                "right": len(navigation_records),
            },
            responses_with_navigation,
        )

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        automation_rows = events.loc[
            events["event_type"].isin(["detector", "result"])
            & events["label"].str.contains("Automation", na=False)
        ].reset_index(drop=True)
        tracker.record("filter", len(events), automation_rows)

        automation_rows = automation_rows.copy()
        automation_rows["event_pair"] = list(
            zip(automation_rows["event_type"], automation_rows["label"])
        )
        automation_events = (
            automation_rows.groupby("incident_id", sort=False)
            .agg(automation_event_labels=("event_pair", list))
            .reset_index()
        )
        tracker.record("groupby", len(automation_rows), automation_events)

        joined = responses_with_navigation.merge(
            automation_events,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(responses_with_navigation),
                "right": len(automation_events),
            },
            joined,
        )

        projected = joined[
            ["incident_id", "narrative_response", "automation_event_labels"]
        ].copy()
        projected["response_pattern"] = projected.apply(
            lambda row: (
                row["narrative_response"]
                if pd.notna(row["narrative_response"])
                else structured_response(row["automation_event_labels"])
            ),
            axis=1,
        )
        projected = projected[["incident_id", "response_pattern"]]
        tracker.record("project", len(joined), projected)

        selected = projected.loc[
            projected["response_pattern"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(projected), selected)

        grouped = (
            selected.groupby("response_pattern", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        response_order = {
            label: index for index, label in enumerate(RESPONSES)
        }
        grouped = grouped.sort_values(
            "response_pattern",
            key=lambda values: values.map(response_order),
        ).reset_index(drop=True)
        tracker.record("groupby", len(selected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
