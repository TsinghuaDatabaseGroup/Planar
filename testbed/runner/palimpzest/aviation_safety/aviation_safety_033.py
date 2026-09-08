#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-033."""

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

TASK_ID = "aviation_safety-033"
DISRUPTION_REASONS = (
    "predeparture_component_discrepancy",
    "mel_or_deferred_item",
    "maintenance_documentation_or_release",
    "inflight_component_or_system_problem",
    "postflight_inspection_or_repair",
)
CLASSIFICATION_LABELS = (*DISRUPTION_REASONS, "not_target")
RESULT_LABELS = (
    "General Maintenance Action",
    "General Flight Cancelled / Delayed",
    "Flight Crew Returned To Departure Airport",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        result_rows = events.loc[
            (events["event_type"] == "result")
            & events["label"].isin(RESULT_LABELS)
        ].reset_index(drop=True)
        tracker.record("filter", len(events), result_rows)

        result_labels = (
            result_rows.groupby("incident_id", sort=False)
            .agg(
                result_labels=(
                    "label",
                    lambda values: list(dict.fromkeys(values)),
                )
            )
            .reset_index()
        )
        tracker.record("groupby", len(result_rows), result_labels)

        assessments = load_table(
            "asrs",
            "assessments.csv",
            ["incident_id", "assessment_type", "label"],
        )
        tracker.record("scan", None, assessments)

        contributing_rows = assessments.loc[
            (assessments["assessment_type"] == "contributing_factor")
            & assessments["label"].isin(["Aircraft", "MEL"])
        ].reset_index(drop=True)
        tracker.record("filter", len(assessments), contributing_rows)

        contributing_labels = (
            contributing_rows.groupby("incident_id", sort=False)
            .agg(
                contributing_labels=(
                    "label",
                    lambda values: list(dict.fromkeys(values)),
                )
            )
            .reset_index()
        )
        tracker.record(
            "groupby", len(contributing_rows), contributing_labels
        )

        label_candidates = result_labels.merge(
            contributing_labels,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(result_labels),
                "right": len(contributing_labels),
            },
            label_candidates,
        )

        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file"],
        )
        tracker.record("scan", None, incidents)

        joined = label_candidates.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {"left": len(label_candidates), "right": len(incidents)},
            joined,
        )

        reports = load_selected_texts("asrs", joined)[
            ["incident_id", "result_labels", "contributing_labels", "text"]
        ]
        tracker.record("scan", len(joined), reports)

        reason_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "disruption_reason",
                    "type": str,
                    "desc": (
                        "Exactly one clearly primary reason from "
                        "predeparture_component_discrepancy, "
                        "mel_or_deferred_item, "
                        "maintenance_documentation_or_release, "
                        "inflight_component_or_system_problem, "
                        "postflight_inspection_or_repair, or not_target when no "
                        "single reason clearly fits."
                    ),
                }
            ],
            desc=(
                "Assign the single primary disruption reason. Distinguish a "
                "component discrepancy found before departure, an MEL or deferred "
                "item, maintenance documentation or release, an in-flight "
                "component or system problem, and post-flight inspection or "
                "repair. Use not_target when the report does not clearly fit one."
            ),
            depends_on=[
                "incident_id",
                "result_labels",
                "contributing_labels",
                "text",
            ],
        )
        started = time.time()
        reason_result = reason_plan.run(config)
        classified = result_frame(reason_result)
        classified["disruption_reason"] = classified[
            "disruption_reason"
        ].map(lambda value: normalize_enum(value, CLASSIFICATION_LABELS))
        tracker.record_semantic(
            "sem_map",
            len(reports),
            classified,
            reason_result,
            time.time() - started,
        )

        selected = classified.loc[
            classified["disruption_reason"].isin(DISRUPTION_REASONS),
            ["incident_id", "disruption_reason"],
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), selected)

        grouped = (
            selected.groupby("disruption_reason", sort=True)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record("groupby", len(selected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
