#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-025."""

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

TASK_ID = "aviation_safety-025"
MECHANISMS = (
    "data_entry_or_programming",
    "alert_interpretation",
    "automation_disengagement",
    "unexpected_capture_or_leveloff",
    "mode_awareness",
)
CLASSIFICATION_LABELS = (*MECHANISMS, "not_target")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem"],
        )
        tracker.record("scan", None, incidents)

        human_factors = incidents.loc[
            incidents["primary_problem"] == "Human Factors"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), human_factors)

        reports = load_selected_texts("asrs", human_factors)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(human_factors), reports)

        contributed_plan = memory_dataset(
            f"{TASK_ID}-contributed", reports
        ).sem_filter(
            (
                "Keep this report only if misunderstanding, incorrect mental "
                "model, or surprise about aircraft automation materially "
                "contributed to the event."
            ),
            depends_on=["text"],
        )
        started = time.time()
        contributed_result = contributed_plan.run(config)
        contributed = result_frame(contributed_result)[
            ["incident_id", "text"]
        ]
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            contributed,
            contributed_result,
            time.time() - started,
        )

        non_equipment_plan = memory_dataset(
            f"{TASK_ID}-non-equipment", contributed
        ).sem_filter(
            (
                "Keep this report only if its primary issue was neither an "
                "aircraft equipment failure nor an ordinary manual-flying "
                "error without automation confusion."
            ),
            depends_on=["text"],
        )
        started = time.time()
        non_equipment_result = non_equipment_plan.run(config)
        eligible = result_frame(non_equipment_result)[
            ["incident_id", "text"]
        ]
        tracker.record_semantic(
            "sem_filter",
            len(contributed),
            eligible,
            non_equipment_result,
            time.time() - started,
        )

        classification_plan = memory_dataset(
            f"{TASK_ID}-classification", eligible
        ).sem_map(
            cols=[
                {
                    "name": "confusion_mechanism",
                    "type": str,
                    "desc": (
                        "Exactly one of data_entry_or_programming, "
                        "alert_interpretation, automation_disengagement, "
                        "unexpected_capture_or_leveloff, mode_awareness, or "
                        "not_target. Use the first applicable category in that "
                        "order. Use not_target if none clearly applies."
                    ),
                }
            ],
            desc=(
                "Assign the dominant automation-confusion mechanism. "
                "data_entry_or_programming concerns entered data or programmed "
                "automation selections; alert_interpretation concerns "
                "misunderstood alerts or annunciations; "
                "automation_disengagement concerns an automation disengagement; "
                "unexpected_capture_or_leveloff concerns an unexpected capture "
                "or level-off; mode_awareness concerns awareness of the active "
                "or armed automation mode. Apply the stated precedence."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(classification_result)
        classified["confusion_mechanism"] = classified[
            "confusion_mechanism"
        ].map(lambda value: normalize_enum(value, CLASSIFICATION_LABELS))
        tracker.record_semantic(
            "sem_map",
            len(eligible),
            classified,
            classification_result,
            time.time() - started,
        )

        selected = classified.loc[
            classified["confusion_mechanism"].isin(MECHANISMS),
            ["incident_id", "confusion_mechanism"],
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), selected)

        grouped = (
            selected.groupby("confusion_mechanism", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        mechanism_order = {
            label: index for index, label in enumerate(MECHANISMS)
        }
        grouped = grouped.sort_values(
            "confusion_mechanism",
            key=lambda values: values.map(mechanism_order),
        ).reset_index(drop=True)
        tracker.record("groupby", len(selected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
