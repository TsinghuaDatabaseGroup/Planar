#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-020."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_jsonl,
    normalize_enum,
    save_output,
    setup,
)


TASK_ID = "aviation_safety-020"
DISRUPTION_REASONS = (
    "predeparture_component_discrepancy",
    "mel_or_deferred_item",
    "maintenance_documentation_or_release",
    "inflight_component_or_system_problem",
    "postflight_inspection_or_repair",
    "not_target",
)


def main():
    setup(max_tokens=384)
    tracker = StepTracker()

    with Timer() as timer:
        candidates = load_jsonl(
            "operator_implement",
            "inputs/OP-SEM_GROUP_BY-003.jsonl",
        )[
            [
                "incident_id",
                "result_labels",
                "contributing_labels",
                "text",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(operator_implement/inputs/OP-SEM_GROUP_BY-003.jsonl)",
            None,
            len(candidates),
            output=candidates,
        )

        with tracker.step(
            "SEM_CLASSIFY(primary disruption reason)",
            input_rows=len(candidates),
        ) as step:
            classified = candidates.sem_map(
                "Assign exactly one primary disruption_reason using the result "
                "labels {result_labels}, contributing labels "
                "{contributing_labels}, and narrative {text}. Choose exactly one "
                "of predeparture_component_discrepancy, mel_or_deferred_item, "
                "maintenance_documentation_or_release, "
                "inflight_component_or_system_problem, "
                "postflight_inspection_or_repair, or not_target when none clearly "
                "fits. Output only the label.",
                suffix="disruption_reason",
            )
            classified["disruption_reason"] = classified[
                "disruption_reason"
            ].map(
                lambda value: normalize_enum(value, DISRUPTION_REASONS)
                or "not_target"
            )
            step.set_output(classified)

        grouped = (
            classified.groupby("disruption_reason", sort=False)
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
            "GROUP_BY(disruption_reason, COUNT_DISTINCT(incident_id))",
            len(classified),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
