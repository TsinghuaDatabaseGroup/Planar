#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-008."""

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

TASK_ID = "aviation_safety-008"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "far_part", "mission"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[
            (incidents["far_part"] == "Part 121")
            & (incidents["mission"] == "Passenger")
        ].copy()
        tracker.record(
            "FILTER(far_part='Part 121' AND mission='Passenger')",
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
            "SEM_FILTER(clearance communication problem corrected before consequence)",
            input_rows=len(reports),
        ) as step:
            hits = reports.sem_filter(
                "The report {text} describes an ATC clearance, readback, or "
                "hearback problem that the crew detected or corrected before it "
                "caused a loss of separation, altitude deviation, runway incursion, "
                "or any other operational consequence"
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
