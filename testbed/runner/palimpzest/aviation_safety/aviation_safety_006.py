#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-006."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-006"


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

        semantic = memory_dataset(TASK_ID, reports).sem_filter(
            (
                "Keep this report only if it describes a taxi, runway-crossing, or "
                "hold-short clearance conflict that was explicitly detected, caught, "
                "or stopped before the threatened aircraft entered the conflicting "
                "runway. The timing and preventive intervention must be supported by "
                "the report; a conflict with no such pre-entry resolution does not qualify."
            ),
            depends_on=["text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        matched = result_frame(semantic_result)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            matched,
            semantic_result,
            time.time() - started,
        )

        answer = (
            int(matched["incident_id"].nunique())
            if "incident_id" in matched.columns
            else 0
        )
        tracker.record(
            "groupby",
            len(matched),
            answer,
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
