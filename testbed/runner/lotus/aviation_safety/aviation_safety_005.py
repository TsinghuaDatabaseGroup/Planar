#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-005."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    load_selected_texts,
    load_table,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-005"


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

        filtered = incidents[incidents["state_reference"] == "US"].copy()
        tracker.record(
            "FILTER(state_reference='US')",
            len(incidents),
            len(filtered),
            output=filtered,
        )

        reports = load_selected_texts("asrs", filtered)[["incident_id", "text"]]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(horizontal and vertical closest-approach separation)",
            input_rows=len(reports),
        ) as step:
            hits = reports.sem_filter(
                "The report {text} states both horizontal separation and vertical "
                "separation for the same closest-approach event, and each dimension "
                "is numerically recoverable in feet using yards = 3 feet, nautical "
                "miles approximately = 6076 feet, and statute miles = 5280 feet"
            )
            step.set_output(hits)

        answer = int(hits["incident_id"].nunique())
        tracker.record(
            "GROUP_BY([], COUNT_DISTINCT(incident_id))",
            len(hits),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
