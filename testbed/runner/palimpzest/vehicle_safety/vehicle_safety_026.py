#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-026."""

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

TASK_ID = "vehicle_safety-026"
DATASET = "nhtsa_vehicle_safety"
ROOT_CAUSE_LABELS = (
    "spontaneous_glass_breakage",
    "door_water_intrusion",
    "visibility_distortion_or_defect",
    "other",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["make", "model", "component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        rav4_body = complaints.loc[
            (complaints["make"] == "TOYOTA")
            & (complaints["model"] == "RAV4")
            & (complaints["component_id"] == "BODY")
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), rav4_body)

        relevance_plan = memory_dataset(TASK_ID, rav4_body).sem_filter(
            filter=(
                "The complaint summary describes a safety-relevant glass, window, "
                "windshield, sunroof or moonroof, or water-in-door/window visibility "
                "defect."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        relevant = result_frame(relevance_result, rav4_body)
        tracker.record_semantic(
            "sem_filter",
            len(rav4_body),
            relevant,
            relevance_result,
            time.time() - started,
        )

        classification_plan = memory_dataset(
            f"{TASK_ID}-root-cause", relevant
        ).sem_map(
            cols=[
                {
                    "name": "root_cause_label",
                    "type": str,
                    "desc": (
                        "Exactly one of spontaneous_glass_breakage, "
                        "door_water_intrusion, visibility_distortion_or_defect, "
                        "or other."
                    ),
                }
            ],
            desc="Classify the likely primary root cause described by the complaint.",
            depends_on=["summary_text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            relevant,
            ["root_cause_label"],
        )
        classified["root_cause_label"] = classified[
            "root_cause_label"
        ].map(lambda value: normalize_enum(value, ROOT_CAUSE_LABELS) or "other")
        tracker.record_semantic(
            "sem_map",
            len(relevant),
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

        if grouped.empty:
            most_common_label = None
        else:
            highest_count = grouped["count"].max()
            most_common_label = sorted(
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
        answer = {
            "most_common_label": most_common_label,
            "per_label_counts": per_label_counts,
        }
        tracker.record("project", len(grouped), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
