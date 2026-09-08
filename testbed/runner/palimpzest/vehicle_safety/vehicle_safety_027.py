#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-027."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_table,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-027"
DATASET = "nhtsa_vehicle_safety"
ROOT_CAUSE_LABELS = (
    "sensor_misfire",
    "software_logic_error",
    "inflator_defect",
    "other",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["make", "component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        toyota_airbag = complaints.loc[
            (complaints["make"] == "TOYOTA")
            & (complaints["component_id"] == "AIRBAG")
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), toyota_airbag)

        deployment_plan = memory_dataset(
            f"{TASK_ID}-deployment", toyota_airbag
        ).sem_filter(
            filter=(
                "The complaint summary describes an unintended airbag deployment "
                "without a corresponding collision or impact."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        deployment_result = deployment_plan.run(config)
        unintended = result_frame(deployment_result, toyota_airbag)
        tracker.record_semantic(
            "sem_filter",
            len(toyota_airbag),
            unintended,
            deployment_result,
            time.time() - started,
        )

        injury_plan = memory_dataset(
            f"{TASK_ID}-injury", unintended
        ).sem_filter(
            filter=(
                "The complaint summary reports an occupant injury or burn caused "
                "by the airbag deployment."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        injury_result = injury_plan.run(config)
        injury_cases = result_frame(injury_result, unintended)
        tracker.record_semantic(
            "sem_filter",
            len(unintended),
            injury_cases,
            injury_result,
            time.time() - started,
        )

        classification_plan = memory_dataset(
            f"{TASK_ID}-root-cause", injury_cases
        ).sem_map(
            cols=[
                {
                    "name": "root_cause_label",
                    "type": str,
                    "desc": (
                        "Exactly one of sensor_misfire, software_logic_error, "
                        "inflator_defect, or other."
                    ),
                }
            ],
            desc="Classify the complaint's likely primary root cause.",
            depends_on=["summary_text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            injury_cases,
            ["root_cause_label"],
        )
        classified["root_cause_label"] = classified[
            "root_cause_label"
        ].map(lambda value: normalize_enum(value, ROOT_CAUSE_LABELS) or "other")
        tracker.record_semantic(
            "sem_map",
            len(injury_cases),
            classified,
            classification_result,
            time.time() - started,
        )

        grouped = (
            classified.groupby(
                "root_cause_label",
                as_index=False,
                dropna=False,
            )
            .size()
            .rename(columns={"size": "count"})
        )
        tracker.record("groupby", len(classified), grouped)

        total = int(grouped["count"].sum())
        if grouped.empty:
            dominant_label = None
        else:
            highest_count = grouped["count"].max()
            dominant_label = sorted(
                grouped.loc[
                    grouped["count"] == highest_count,
                    "root_cause_label",
                ].astype(str)
            )[0]
        per_label_counts = {
            str(label): {"count": int(count)}
            for label, count in zip(
                grouped["root_cause_label"],
                grouped["count"],
                strict=True,
            )
        }
        per_label_percentages = (
            {
                str(label): round(100.0 * int(count) / total, 1)
                for label, count in zip(
                    grouped["root_cause_label"],
                    grouped["count"],
                    strict=True,
                )
            }
            if total
            else {}
        )
        answer = {
            "dominant_label": dominant_label,
            "per_label_counts": per_label_counts,
            "per_label_percentages": per_label_percentages,
        }
        tracker.record("project", len(grouped), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
