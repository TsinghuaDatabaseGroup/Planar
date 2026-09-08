#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-034."""

from __future__ import annotations

import os
import re
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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-034"
PATTERNS = (
    "direct_conflict",
    "modified_or_delayed_compliance",
    "clarification_required",
)
SEVERITY_ORDER = {label: index for index, label in enumerate(PATTERNS, start=1)}
AUTOMATION_DETECTORS = ("Automation Aircraft RA", "Automation Aircraft TA")


def parse_miss_distance(label) -> tuple[str | None, int | None]:
    match = re.match(
        r"^(Horizontal|Vertical)\s+(-?\d+)", str(label).strip()
    )
    if not match:
        return None, None
    return match.group(1).lower(), int(match.group(2))


def min_for_direction(
    values: pd.Series,
    frame: pd.DataFrame,
    direction: str,
):
    selected = values.loc[
        frame.loc[values.index, "distance_type"] == direction
    ].dropna()
    return selected.min() if not selected.empty else pd.NA


def collect_detectors(values: pd.Series, frame: pd.DataFrame) -> list[str]:
    selected = values.loc[
        frame.loc[values.index, "event_type"] == "detector"
    ]
    return list(dict.fromkeys(selected))


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

        conflict_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "qualifies",
                    "type": bool,
                    "desc": (
                        "True only when automation guidance and an ATC "
                        "instruction were difficult to reconcile in the same "
                        "episode."
                    ),
                },
                {
                    "name": "reconciliation_pattern",
                    "type": str,
                    "desc": (
                        "For a qualifying report, exactly one of direct_conflict, "
                        "modified_or_delayed_compliance, or clarification_required."
                    ),
                },
                {
                    "name": "automation_guidance",
                    "type": str,
                    "desc": (
                        "The relevant automation guidance in no more than eight "
                        "words."
                    ),
                },
                {
                    "name": "atc_instruction",
                    "type": str,
                    "desc": (
                        "The conflicting or difficult-to-reconcile ATC instruction "
                        "in no more than eight words."
                    ),
                },
            ],
            desc=(
                "Identify a same-episode conflict between automation guidance and "
                "an ATC instruction. Classify a direct incompatibility as "
                "direct_conflict, compliance that was modified or delayed as "
                "modified_or_delayed_compliance, and a situation resolved or "
                "addressed through clarification as clarification_required."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        conflict_result = conflict_plan.run(config)
        extracted = result_frame(conflict_result)
        extracted["qualifies"] = extracted["qualifies"].map(parse_bool)
        extracted["reconciliation_pattern"] = extracted[
            "reconciliation_pattern"
        ].map(lambda value: normalize_enum(value, PATTERNS))
        for column in ("automation_guidance", "atc_instruction"):
            extracted[column] = extracted[column].fillna("").astype(str).str.strip()
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            conflict_result,
            time.time() - started,
        )

        semantic_conflicts = extracted.loc[
            extracted["qualifies"]
            & extracted["reconciliation_pattern"].notna()
            & extracted["automation_guidance"].ne("")
            & extracted["atc_instruction"].ne(""),
            [
                "incident_id",
                "reconciliation_pattern",
                "automation_guidance",
                "atc_instruction",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), semantic_conflicts)

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        event_rows = events.loc[
            (
                (events["event_type"] == "detector")
                & events["label"].isin(AUTOMATION_DETECTORS)
            )
            | (events["event_type"] == "miss_distance")
        ].reset_index(drop=True)
        tracker.record("filter", len(events), event_rows)

        event_rows = event_rows.copy()
        parsed = event_rows["label"].map(parse_miss_distance)
        event_rows["distance_type"] = parsed.map(lambda value: value[0])
        event_rows["distance_ft"] = parsed.map(lambda value: value[1])
        event_records = (
            event_rows.groupby("incident_id", sort=False)
            .agg(
                automation_detectors=(
                    "label",
                    lambda values: collect_detectors(values, event_rows),
                ),
                horizontal_separation_ft=(
                    "distance_ft",
                    lambda values: min_for_direction(
                        values, event_rows, "horizontal"
                    ),
                ),
                vertical_separation_ft=(
                    "distance_ft",
                    lambda values: min_for_direction(
                        values, event_rows, "vertical"
                    ),
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(event_rows), event_records)

        complete_events = event_records.loc[
            event_records["automation_detectors"].map(bool)
            & event_records["horizontal_separation_ft"].notna()
            & event_records["vertical_separation_ft"].notna()
        ].reset_index(drop=True)
        for column in ("horizontal_separation_ft", "vertical_separation_ft"):
            complete_events[column] = complete_events[column].astype(int)
        tracker.record("filter", len(event_records), complete_events)

        joined = semantic_conflicts.merge(
            complete_events,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(semantic_conflicts),
                "right": len(complete_events),
            },
            joined,
        )

        projected = joined[
            [
                "incident_id",
                "reconciliation_pattern",
                "automation_guidance",
                "atc_instruction",
                "horizontal_separation_ft",
                "vertical_separation_ft",
            ]
        ].copy()
        projected["severity_order"] = projected[
            "reconciliation_pattern"
        ].map(SEVERITY_ORDER)
        tracker.record("project", len(joined), projected)

        ordered = projected.sort_values(
            [
                "severity_order",
                "horizontal_separation_ft",
                "vertical_separation_ft",
                "incident_id",
            ],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
        tracker.record("sort", len(projected), ordered)

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited[
            [
                "incident_id",
                "reconciliation_pattern",
                "automation_guidance",
                "atc_instruction",
                "horizontal_separation_ft",
                "vertical_separation_ft",
            ]
        ].copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
