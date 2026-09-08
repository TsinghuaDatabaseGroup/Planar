#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-005."""

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

TASK_ID = "aviation_safety-005"


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

        filtered = incidents.loc[
            incidents["state_reference"] == "US"
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
                "Keep this report only if it states both a quantitative horizontal "
                "separation and a quantitative vertical separation for the same "
                "closest-approach event, and both dimensions can be recovered in feet. "
                "Use yards = 3 feet, nautical miles approximately 6076 feet, and "
                "statute miles = 5280 feet. Do not combine measurements from different "
                "events or accept a dimension that has no recoverable numeric value."
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
