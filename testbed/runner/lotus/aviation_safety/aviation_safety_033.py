#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-033."""

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

TASK_ID = "aviation_safety-033"
TARGET_RESULTS = {
    "General Maintenance Action",
    "General Flight Cancelled / Delayed",
    "Flight Crew Returned To Departure Airport",
}
TARGET_CONTRIBUTORS = {"Aircraft", "MEL"}
DISRUPTION_REASONS = (
    "predeparture_component_discrepancy",
    "mel_or_deferred_item",
    "maintenance_documentation_or_release",
    "inflight_component_or_system_problem",
    "postflight_inspection_or_repair",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        result_rows = events[
            (events["event_type"] == "result")
            & events["label"].isin(TARGET_RESULTS)
        ].copy()
        tracker.record(
            "FILTER(event_type='result' AND label IN target results)",
            len(events),
            len(result_rows),
            output=result_rows,
        )

        result_labels = (
            result_rows.groupby("incident_id", sort=False)
            .agg(result_labels=("label", list))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(result_labels))",
            len(result_rows),
            len(result_labels),
            output=result_labels,
        )

        assessments = load_table("asrs", "assessments.csv")[
            ["incident_id", "assessment_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(assessments.csv)",
            None,
            len(assessments),
            output=assessments,
        )

        contributing_rows = assessments[
            (assessments["assessment_type"] == "contributing_factor")
            & assessments["label"].isin(TARGET_CONTRIBUTORS)
        ].copy()
        tracker.record(
            "FILTER(contributing_factor AND label IN Aircraft/MEL)",
            len(assessments),
            len(contributing_rows),
            output=contributing_rows,
        )

        contributing_labels = (
            contributing_rows.groupby("incident_id", sort=False)
            .agg(contributing_labels=("label", list))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(contributing_labels))",
            len(contributing_rows),
            len(contributing_labels),
            output=contributing_labels,
        )

        label_candidates = result_labels.merge(
            contributing_labels,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(result_labels, contributing_labels, incident_id)",
            {
                "left": len(result_labels),
                "right": len(contributing_labels),
            },
            len(label_candidates),
            output=label_candidates,
        )

        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        joined = label_candidates.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(label_candidates, incidents, incident_id)",
            {"left": len(label_candidates), "right": len(incidents)},
            len(joined),
            output=joined,
        )

        reports = load_selected_texts("asrs", joined)[
            ["incident_id", "result_labels", "contributing_labels", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=joined.text_file)",
            len(joined),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(primary disruption reason)",
            input_rows=len(reports),
        ) as step:
            classified = reports.sem_extract(
                input_cols=["result_labels", "contributing_labels", "text"],
                output_cols={
                    "disruption_reason": (
                        "assign exactly one clearly primary reason from "
                        "predeparture_component_discrepancy, mel_or_deferred_item, "
                        "maintenance_documentation_or_release, "
                        "inflight_component_or_system_problem, "
                        "postflight_inspection_or_repair; return not_applicable when "
                        "the report does not clearly fit one"
                    )
                },
            )
            classified["disruption_reason"] = classified[
                "disruption_reason"
            ].map(lambda value: normalize_enum(value, DISRUPTION_REASONS))
            assigned = classified.loc[
                classified["disruption_reason"].notna(),
                ["incident_id", "disruption_reason"],
            ].reset_index(drop=True)
            step.set_output(assigned)

        grouped = (
            assigned.groupby("disruption_reason", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        reason_order = {
            label: index for index, label in enumerate(DISRUPTION_REASONS)
        }
        grouped = grouped.sort_values(
            "disruption_reason",
            key=lambda values: values.map(reason_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([disruption_reason], COUNT_DISTINCT)",
            len(assigned),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
