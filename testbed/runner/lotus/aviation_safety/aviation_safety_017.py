#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-017."""

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


TASK_ID = "aviation_safety-017"


def main():
    setup(max_tokens=256)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "text_file",
                "anomaly_summary",
                "result_summary",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        equipment_reports = incidents[
            incidents["anomaly_summary"].str.contains(
                "Aircraft Equipment Problem",
                case=False,
                na=False,
                regex=False,
            )
        ].copy()
        tracker.record(
            "FILTER(anomaly_summary CONTAINS 'Aircraft Equipment Problem')",
            len(incidents),
            len(equipment_reports),
            output=equipment_reports,
        )

        reports = load_selected_texts("asrs", equipment_reports)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(equipment_reports),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(active troubleshooting with classifiable resulting status)",
            input_rows=len(reports),
        ) as step:
            qualifying = reports.sem_filter(
                "Keep this report when {text}, together with its structured result "
                "summary {result_summary}, describes active crew or maintenance "
                "troubleshooting and supports a clear resulting status: resolved "
                "when corrective action cleared the problem, intermittent when "
                "the problem disappeared but remained uncertain or recurred, or "
                "still unsafe when it persisted or left the aircraft unsafe."
            )
            step.set_output(qualifying)

        answer = int(qualifying["incident_id"].nunique())
        tracker.record(
            "GROUP_BY([], COUNT_DISTINCT(incident_id))",
            len(qualifying),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
