#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-012."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-012"
STATUSES = ("still_unsafe", "intermittent", "resolved")
STATUS_LABELS = (*STATUSES, "not_target")


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "anomaly_summary", "result_summary"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        equipment = incidents[
            incidents["anomaly_summary"].str.contains(
                "Aircraft Equipment Problem",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(anomaly_summary CONTAINS 'Aircraft Equipment Problem')",
            len(incidents),
            len(equipment),
            output=equipment,
        )

        reports = load_selected_texts("asrs", equipment)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(equipment),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(assign post-troubleshooting status)",
            input_rows=len(reports),
        ) as step:
            classified = reports.sem_extract(
                input_cols=["result_summary", "text"],
                output_cols={
                    "post_troubleshooting_status": (
                        "assign exactly one status using this precedence: "
                        "still_unsafe when active crew or maintenance "
                        "troubleshooting left the problem persistent or the "
                        "aircraft unsafe; intermittent when the problem disappeared "
                        "but remained uncertain or later recurred; resolved when "
                        "corrective action clearly cleared it; not_target when no "
                        "active troubleshooting and resulting status are described. "
                        "A structured result that the equipment problem dissipated "
                        "may establish resolved only when consistent with the "
                        "narrative"
                    )
                },
            )
            classified["post_troubleshooting_status"] = classified[
                "post_troubleshooting_status"
            ].map(lambda value: normalize_enum(value, STATUS_LABELS))
            classified["post_troubleshooting_status"] = classified[
                "post_troubleshooting_status"
            ].fillna("not_target")
            classified = classified[
                ["incident_id", "post_troubleshooting_status"]
            ].reset_index(drop=True)
            step.set_output(classified)

        targeted = classified[
            classified["post_troubleshooting_status"] != "not_target"
        ].copy()
        tracker.record(
            "FILTER(post_troubleshooting_status != 'not_target')",
            len(classified),
            len(targeted),
            output=targeted,
        )

        grouped = (
            targeted.groupby("post_troubleshooting_status", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        status_order = {label: index for index, label in enumerate(STATUSES)}
        grouped = grouped.sort_values(
            "post_troubleshooting_status",
            key=lambda values: values.map(status_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([post_troubleshooting_status], COUNT_DISTINCT)",
            len(targeted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
