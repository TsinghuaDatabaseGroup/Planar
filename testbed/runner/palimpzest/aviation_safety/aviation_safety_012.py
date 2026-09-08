#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-012."""

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

TASK_ID = "aviation_safety-012"
STATUSES = ("still_unsafe", "intermittent", "resolved")
CLASSIFICATION_LABELS = (*STATUSES, "not_target")


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
                "anomaly_summary",
                "result_summary",
            ],
        )
        tracker.record("scan", None, incidents)

        equipment = incidents.loc[
            incidents["anomaly_summary"].str.contains(
                "Aircraft Equipment Problem", na=False
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), equipment)

        reports = load_selected_texts("asrs", equipment)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record("scan", len(equipment), reports)

        classification_plan = memory_dataset(
            TASK_ID, reports
        ).sem_map(
            cols=[
                {
                    "name": "post_troubleshooting_status",
                    "type": str,
                    "desc": (
                        "Exactly one of still_unsafe, intermittent, resolved, or "
                        "not_target. Use still_unsafe when the problem persisted "
                        "or left the aircraft unsafe; intermittent when it "
                        "disappeared but remained uncertain or later recurred; "
                        "resolved when corrective action clearly cleared it; and "
                        "not_target when active crew or maintenance troubleshooting "
                        "and a resulting status are not described. Apply this "
                        "precedence in the listed order."
                    ),
                }
            ],
            desc=(
                "Classify the status after active crew or maintenance "
                "troubleshooting. A structured result stating that the equipment "
                "problem dissipated may establish resolved only when the narrative "
                "is consistent with that result."
            ),
            depends_on=["incident_id", "result_summary", "text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(classification_result)
        classified["post_troubleshooting_status"] = classified[
            "post_troubleshooting_status"
        ].map(lambda value: normalize_enum(value, CLASSIFICATION_LABELS))
        tracker.record_semantic(
            "sem_map",
            len(reports),
            classified,
            classification_result,
            time.time() - started,
        )

        selected = classified.loc[
            classified["post_troubleshooting_status"].isin(STATUSES),
            ["incident_id", "post_troubleshooting_status"],
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), selected)

        grouped = (
            selected.groupby("post_troubleshooting_status", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        status_order = {label: index for index, label in enumerate(STATUSES)}
        grouped = grouped.sort_values(
            "post_troubleshooting_status",
            key=lambda values: values.map(status_order),
        ).reset_index(drop=True)
        tracker.record("groupby", len(selected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
