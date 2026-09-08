#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-030."""

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
    parse_number,
    parse_number_after,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-030"


def minimum_distance(labels, prefix):
    values = [
        parsed
        for label in labels
        if str(label).startswith(prefix)
        and (parsed := parse_number_after(label, prefix)) is not None
    ]
    return min(values) if values else None


def comparison_label(narrative_value, structured_value) -> str:
    if pd.isna(narrative_value):
        return "not_stated"
    return "matches" if narrative_value == structured_value else "differs"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "state_reference"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        us_incidents = incidents[incidents["state_reference"] == "US"].copy()
        tracker.record(
            "FILTER(state_reference='US')",
            len(incidents),
            len(us_incidents),
            output=us_incidents,
        )

        reports = load_selected_texts("asrs", us_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(us_incidents),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(conflict parties and narrative miss distances)",
            input_rows=len(reports),
        ) as step:
            semantic_distances = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "parties_involved": (
                        "the parties involved in the closest conflict as a natural "
                        "phrase of at most six words"
                    ),
                    "narrative_horizontal_ft": (
                        "the explicitly stated narrative horizontal closest "
                        "separation converted to an integer number of feet; null if "
                        "the horizontal dimension is not stated"
                    ),
                    "narrative_vertical_ft": (
                        "the explicitly stated narrative vertical closest separation "
                        "converted to an integer number of feet; null if the vertical "
                        "dimension is not stated"
                    ),
                },
            )
            semantic_distances["parties_involved"] = semantic_distances[
                "parties_involved"
            ].map(clean_text)
            semantic_distances["narrative_horizontal_ft"] = semantic_distances[
                "narrative_horizontal_ft"
            ].map(parse_number)
            semantic_distances["narrative_vertical_ft"] = semantic_distances[
                "narrative_vertical_ft"
            ].map(parse_number)
            semantic_distances = semantic_distances[
                [
                    "incident_id",
                    "parties_involved",
                    "narrative_horizontal_ft",
                    "narrative_vertical_ft",
                ]
            ].reset_index(drop=True)
            step.set_output(semantic_distances)

        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        miss_distances = events[events["event_type"] == "miss_distance"].copy()
        tracker.record(
            "FILTER(event_type='miss_distance')",
            len(events),
            len(miss_distances),
            output=miss_distances,
        )

        structured_distances = (
            miss_distances.groupby("incident_id", sort=False)
            .agg(
                structured_horizontal_ft=(
                    "label",
                    lambda labels: minimum_distance(labels, "Horizontal "),
                ),
                structured_vertical_ft=(
                    "label",
                    lambda labels: minimum_distance(labels, "Vertical "),
                ),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], MIN_IF(horizontal), MIN_IF(vertical))",
            len(miss_distances),
            len(structured_distances),
            output=structured_distances,
        )

        complete_distances = structured_distances[
            structured_distances["structured_horizontal_ft"].notna()
            & structured_distances["structured_vertical_ft"].notna()
        ].copy()
        complete_distances["structured_horizontal_ft"] = complete_distances[
            "structured_horizontal_ft"
        ].astype(int)
        complete_distances["structured_vertical_ft"] = complete_distances[
            "structured_vertical_ft"
        ].astype(int)
        tracker.record(
            "FILTER(structured horizontal and vertical distances IS NOT NULL)",
            len(structured_distances),
            len(complete_distances),
            output=complete_distances,
        )

        joined = semantic_distances.merge(
            complete_distances,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(semantic_distances, structured_distances, incident_id)",
            {
                "left": len(semantic_distances),
                "right": len(complete_distances),
            },
            len(joined),
            output=joined,
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
        projected["horizontal_comparison"] = [
            comparison_label(narrative, structured)
            for narrative, structured in zip(
                projected["narrative_horizontal_ft"],
                projected["structured_horizontal_ft"],
                strict=True,
            )
        ]
        projected["vertical_comparison"] = [
            comparison_label(narrative, structured)
            for narrative, structured in zip(
                projected["narrative_vertical_ft"],
                projected["structured_vertical_ft"],
                strict=True,
            )
        ]
        tracker.record(
            "PROJECT(structured/narrative distances and comparison CASE)",
            len(joined),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            [
                "structured_horizontal_ft",
                "structured_vertical_ft",
                "incident_id",
            ],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([horizontal ASC, vertical ASC, incident_id ASC])",
            len(projected),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(20).copy()
        tracker.record(
            "LIMIT(20)",
            len(ordered),
            len(limited),
            output=limited,
        )

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and distance comparison fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
