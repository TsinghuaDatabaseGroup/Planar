#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-042."""

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
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "aviation_safety-042"
MATCH_COLUMNS = [
    "incident_id",
    "narrative_component_term",
    "first_detection_cue",
    "canonical_component",
]


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

        equipment_rows = events.loc[
            events["label"].str.contains(
                "Aircraft Equipment Problem", regex=False, na=False
            ),
            ["incident_id"],
        ].reset_index(drop=True)
        tracker.record("filter", len(events), equipment_rows)

        equipment_events = equipment_rows.drop_duplicates(
            subset=["incident_id"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(equipment_rows), equipment_events)

        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file"],
        )
        tracker.record("scan", None, incidents)

        equipment_incidents = equipment_events.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {"left": len(equipment_events), "right": len(incidents)},
            equipment_incidents,
        )

        reports = load_selected_texts("asrs", equipment_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(equipment_incidents), reports)

        component_plan = memory_dataset(TASK_ID, reports).sem_flat_map(
            cols=[
                {
                    "name": "narrative_component_term",
                    "type": str,
                    "desc": "The aircraft-component term as used in the report.",
                },
                {
                    "name": "first_detection_cue",
                    "type": str,
                    "desc": (
                        "The first cue by which the problem was detected, in no "
                        "more than six words."
                    ),
                },
            ],
            desc=(
                "Emit one row for each aircraft component actually involved in "
                "the reported equipment problem. Do not emit merely mentioned "
                "or unaffected components."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        component_result = component_plan.run(config)
        narrative_components = result_frame(
            component_result,
            reports,
            ["narrative_component_term", "first_detection_cue"],
        )
        tracker.record_semantic(
            "sem_flat_map",
            len(reports),
            narrative_components,
            component_result,
            time.time() - started,
        )

        components = load_table(
            "asrs",
            "components.csv",
            ["attribute", "value"],
        )
        tracker.record("scan", None, components)

        vocabulary_rows = components.loc[
            components["attribute"] == "aircraft_component", ["value"]
        ].reset_index(drop=True)
        tracker.record("filter", len(components), vocabulary_rows)

        component_vocabulary = (
            vocabulary_rows.drop_duplicates(subset=["value"])
            .rename(columns={"value": "canonical_component"})
            .reset_index(drop=True)
        )
        tracker.record("distinct", len(vocabulary_rows), component_vocabulary)

        left_components = memory_dataset(
            f"{TASK_ID}-narrative-components", narrative_components
        )
        right_components = memory_dataset(
            f"{TASK_ID}-component-vocabulary", component_vocabulary
        )
        match_plan = left_components.sem_join(
            right_components,
            condition=(
                "The narrative component term and canonical component label "
                "denote the same component or a clear part-to-system relationship. "
                "Exact spelling is not required, but generic topical similarity "
                "is insufficient."
            ),
            depends_on=MATCH_COLUMNS,
        )
        started = time.time()
        match_result = match_plan.run(config)
        semantic_component_matches = result_frame(match_result)
        if semantic_component_matches.empty:
            semantic_component_matches = pd.DataFrame(columns=MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {
                "left": len(narrative_components),
                "right": len(component_vocabulary),
            },
            semantic_component_matches,
            match_result,
            time.time() - started,
        )

        grouped = (
            semantic_component_matches.groupby(
                ["canonical_component", "narrative_component_term"],
                sort=False,
            )
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_detection_cue=(
                    "first_detection_cue",
                    stable_mode,
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(semantic_component_matches), grouped)

        ordered = grouped.sort_values(
            [
                "incident_count",
                "canonical_component",
                "narrative_component_term",
            ],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("sort", len(grouped), ordered)

        limited = ordered.head(20).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited.copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
