#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-011."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_jsonl,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "vehicle_safety-011"
DATASET = "nhtsa_vehicle_safety"
ACTION_STYLES = (
    "software_update",
    "part_replacement",
    "dealer_inspection_only",
    "other",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "campaign_manufacturer",
                "campaign_report_received_date",
                "component_component_id",
                "remedy_corrective_action",
            ]
        ]
        tracker.record("scan", None, recalls)

        report_dates = pd.to_datetime(
            recalls["campaign_report_received_date"],
            errors="coerce",
        )
        in_range = recalls.loc[
            report_dates.between(
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2025-12-31"),
                inclusive="both",
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), in_range)

        deduplicated = in_range.drop_duplicates(
            subset=["campaign_number"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(in_range), deduplicated)

        classification_plan = memory_dataset(TASK_ID, deduplicated).sem_map(
            cols=[
                {
                    "name": "corrective_action_style",
                    "type": str,
                    "desc": (
                        "Exactly one of software_update, part_replacement, "
                        "dealer_inspection_only, or other. Use "
                        "dealer_inspection_only only when inspection is the "
                        "corrective action without a software update or part "
                        "replacement."
                    ),
                }
            ],
            desc="Classify the dominant style of the corrective action.",
            depends_on=["remedy_corrective_action"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            deduplicated,
            ["corrective_action_style"],
        )
        classified["corrective_action_style"] = classified[
            "corrective_action_style"
        ].map(lambda value: normalize_enum(value, ACTION_STYLES) or "other")
        tracker.record_semantic(
            "sem_map",
            len(deduplicated),
            classified,
            classification_result,
            time.time() - started,
        )

        grouped = (
            classified.groupby(
                "campaign_manufacturer",
                as_index=False,
                dropna=False,
            )
            .agg(
                recall_count=("campaign_number", "nunique"),
                dominant_action_style=(
                    "corrective_action_style",
                    stable_mode,
                ),
                distinct_components_recalled=(
                    "component_component_id",
                    "nunique",
                ),
            )
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(classified), grouped)

        result = grouped.rename(
            columns={"campaign_manufacturer": "manufacturer"}
        )[
            [
                "manufacturer",
                "recall_count",
                "dominant_action_style",
                "distinct_components_recalled",
            ]
        ].copy()
        tracker.record("project", len(grouped), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
