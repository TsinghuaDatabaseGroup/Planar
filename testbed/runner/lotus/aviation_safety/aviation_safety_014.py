#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-014."""

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

TASK_ID = "aviation_safety-014"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "state_reference", "result_summary"]
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

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(no diversion or return and continued as planned)",
            input_rows=len(reports),
        ) as step:
            continued = reports.sem_filter(
                "The report {text} explicitly states that the flight neither "
                "diverted nor returned and instead continued to the planned "
                "destination or completed a normal landing"
            )
            step.set_output(continued)

        emergencies = continued[
            continued["result_summary"].str.contains(
                "General Declared Emergency",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(result_summary CONTAINS 'General Declared Emergency')",
            len(continued),
            len(emergencies),
            output=emergencies,
        )

        answer = int(emergencies["incident_id"].nunique())
        tracker.record(
            "GROUP_BY([], COUNT_DISTINCT(incident_id))",
            len(emergencies),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
