#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-030."""

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
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-030"


def optional_integer(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return int(round(float(match.group()))) if match else None


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


def comparison(narrative_value, structured_value) -> str:
    if narrative_value is None or pd.isna(narrative_value):
        return "not_stated"
    return "matches" if int(narrative_value) == int(structured_value) else "differs"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "state_reference"],
        )
        tracker.record("scan", None, incidents)

        us_incidents = incidents.loc[
            incidents["state_reference"] == "US"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), us_incidents)

        reports = load_selected_texts("asrs", us_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(us_incidents), reports)

        distance_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "parties_involved",
                    "type": str,
                    "desc": (
                        "The parties involved in the closest conflict, expressed "
                        "as a phrase of no more than six words."
                    ),
                },
                {
                    "name": "narrative_horizontal_ft",
                    "type": int | None,
                    "desc": (
                        "The explicitly stated narrative horizontal closest-"
                        "separation value converted to feet, or null when that "
                        "dimension is not stated."
                    ),
                },
                {
                    "name": "narrative_vertical_ft",
                    "type": int | None,
                    "desc": (
                        "The explicitly stated narrative vertical closest-"
                        "separation value converted to feet, or null when that "
                        "dimension is not stated."
                    ),
                },
            ],
            desc=(
                "Identify the parties in the closest conflict and extract only "
                "explicitly stated horizontal and vertical closest-separation "
                "values. Convert stated units to feet; do not infer an unstated "
                "dimension or combine values from different conflicts."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        distance_result = distance_plan.run(config)
        semantic_distances = result_frame(distance_result)
        semantic_distances["narrative_horizontal_ft"] = semantic_distances[
            "narrative_horizontal_ft"
        ].map(optional_integer)
        semantic_distances["narrative_vertical_ft"] = semantic_distances[
            "narrative_vertical_ft"
        ].map(optional_integer)
        semantic_distances = semantic_distances[
            [
                "incident_id",
                "parties_involved",
                "narrative_horizontal_ft",
                "narrative_vertical_ft",
            ]
        ]
        tracker.record_semantic(
            "sem_map",
            len(reports),
            semantic_distances,
            distance_result,
            time.time() - started,
        )

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        miss_rows = events.loc[
            events["event_type"] == "miss_distance"
        ].reset_index(drop=True)
        tracker.record("filter", len(events), miss_rows)

        miss_rows = miss_rows.copy()
        parsed = miss_rows["label"].map(parse_miss_distance)
        miss_rows["distance_type"] = parsed.map(lambda value: value[0])
        miss_rows["distance_ft"] = parsed.map(lambda value: value[1])
        structured_distances = (
            miss_rows.groupby("incident_id", sort=False)
            .agg(
                structured_horizontal_ft=(
                    "distance_ft",
                    lambda values: min_for_direction(
                        values, miss_rows, "horizontal"
                    ),
                ),
                structured_vertical_ft=(
                    "distance_ft",
                    lambda values: min_for_direction(
                        values, miss_rows, "vertical"
                    ),
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(miss_rows), structured_distances)

        complete_distances = structured_distances.loc[
            structured_distances["structured_horizontal_ft"].notna()
            & structured_distances["structured_vertical_ft"].notna()
        ].reset_index(drop=True)
        for column in ("structured_horizontal_ft", "structured_vertical_ft"):
            complete_distances[column] = complete_distances[column].astype(int)
        tracker.record(
            "filter", len(structured_distances), complete_distances
        )

        joined = semantic_distances.merge(
            complete_distances,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(semantic_distances),
                "right": len(complete_distances),
            },
            joined,
        )

        projected = joined[
            [
                "incident_id",
                "parties_involved",
                "structured_horizontal_ft",
                "structured_vertical_ft",
                "narrative_horizontal_ft",
                "narrative_vertical_ft",
            ]
        ].copy()
        projected["horizontal_comparison"] = projected.apply(
            lambda row: comparison(
                row["narrative_horizontal_ft"],
                row["structured_horizontal_ft"],
            ),
            axis=1,
        )
        projected["vertical_comparison"] = projected.apply(
            lambda row: comparison(
                row["narrative_vertical_ft"],
                row["structured_vertical_ft"],
            ),
            axis=1,
        )
        tracker.record("project", len(joined), projected)

        ordered = projected.sort_values(
            ["structured_horizontal_ft", "structured_vertical_ft", "incident_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        tracker.record("sort", len(projected), ordered)

        limited = ordered.head(20).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited.copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
