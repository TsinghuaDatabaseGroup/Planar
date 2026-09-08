#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-022."""

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
    load_table,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-022"
DATASET = "nhtsa_vehicle_safety"
SAFETY_TOPICS = (
    "steering",
    "braking",
    "airbag",
    "electrical",
    "powertrain",
    "other",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["medical_attention_flag", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        medical = complaints.loc[
            complaints["medical_attention_flag"]
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), medical)

        relevance_plan = memory_dataset(TASK_ID, medical).sem_filter(
            filter=(
                "The complaint summary describes a concrete vehicle malfunction, "
                "crash sequence, or safety incident rather than only a billing, "
                "paperwork, or service-scheduling issue."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        concrete = result_frame(relevance_result, medical)
        tracker.record_semantic(
            "sem_filter",
            len(medical),
            concrete,
            relevance_result,
            time.time() - started,
        )

        classification_plan = memory_dataset(
            f"{TASK_ID}-topic", concrete
        ).sem_map(
            cols=[
                {
                    "name": "topic",
                    "type": str,
                    "desc": (
                        "Exactly one of steering, braking, airbag, electrical, "
                        "powertrain, or other."
                    ),
                }
            ],
            desc="Classify the complaint's primary safety-event topic.",
            depends_on=["summary_text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            concrete,
            ["topic"],
        )
        classified["topic"] = classified["topic"].map(
            lambda value: normalize_enum(value, SAFETY_TOPICS) or "other"
        )
        tracker.record_semantic(
            "sem_map",
            len(concrete),
            classified,
            classification_result,
            time.time() - started,
        )

        grouped = (
            classified.groupby("topic", as_index=False, dropna=False)
            .size()
            .rename(columns={"size": "complaint_count"})
        )
        tracker.record("groupby", len(classified), grouped)

        ordered = grouped.sort_values(
            ["complaint_count", "topic"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)

        top_three = ordered.head(3).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_three)
        answer = df_records(top_three[["topic", "complaint_count"]])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
