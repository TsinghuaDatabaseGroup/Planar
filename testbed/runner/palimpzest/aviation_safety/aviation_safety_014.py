#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-014."""

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

TASK_ID = "aviation_safety-014"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "state_reference", "result_summary"],
        )
        tracker.record("scan", None, incidents)

        us_reports = incidents.loc[
            incidents["state_reference"] == "US"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), us_reports)

        reports = load_selected_texts("asrs", us_reports)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record("scan", len(us_reports), reports)

        continuation_plan = memory_dataset(TASK_ID, reports).sem_filter(
            (
                "Keep this report only if it explicitly states that the flight "
                "neither diverted nor returned and instead continued to the "
                "planned destination or completed a normal landing."
            ),
            depends_on=["text"],
        )
        started = time.time()
        continuation_result = continuation_plan.run(config)
        continued = result_frame(continuation_result)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            continued,
            continuation_result,
            time.time() - started,
        )

        emergencies = continued.loc[
            continued["result_summary"].str.contains(
                "General Declared Emergency", na=False
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(continued), emergencies)

        answer = int(emergencies["incident_id"].nunique())
        tracker.record("groupby", len(emergencies), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
