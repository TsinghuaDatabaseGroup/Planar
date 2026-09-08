#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-017."""

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

TASK_ID = "vehicle_safety-017"
DATASET = "nhtsa_vehicle_safety"
DEPLOYMENT_TIMINGS = (
    "pre_impact_or_no_collision",
    "during_or_after_collision",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        airbag_complaints = complaints.loc[
            complaints["component_id"] == "AIRBAG"
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), airbag_complaints)

        relevance_plan = memory_dataset(TASK_ID, airbag_complaints).sem_filter(
            filter=(
                "The complaint summary describes an unintended or unexpected "
                "airbag deployment."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        unintended = result_frame(relevance_result, airbag_complaints)
        tracker.record_semantic(
            "sem_filter",
            len(airbag_complaints),
            unintended,
            relevance_result,
            time.time() - started,
        )

        classification_plan = memory_dataset(
            f"{TASK_ID}-timing", unintended
        ).sem_map(
            cols=[
                {
                    "name": "deployment_timing",
                    "type": str,
                    "desc": (
                        "Exactly pre_impact_or_no_collision when deployment "
                        "happened before any collision impact or without a "
                        "collision; otherwise exactly during_or_after_collision "
                        "when deployment happened during or after a crash impact."
                    ),
                }
            ],
            desc="Classify when the unintended airbag deployment happened.",
            depends_on=["summary_text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            unintended,
            ["deployment_timing"],
        )
        classified["deployment_timing"] = classified["deployment_timing"].map(
            lambda value: normalize_enum(value, DEPLOYMENT_TIMINGS)
            or "during_or_after_collision"
        )
        tracker.record_semantic(
            "sem_map",
            len(unintended),
            classified,
            classification_result,
            time.time() - started,
        )

        total = len(classified)
        if total == 0:
            raise ValueError(f"{TASK_ID}: ratio denominator is zero")
        pre_impact_count = int(
            classified["deployment_timing"].eq(
                "pre_impact_or_no_collision"
            ).sum()
        )
        tracker.record("groupby", len(classified), [total, pre_impact_count])

        answer = round(pre_impact_count / total, 3)
        tracker.record("project", 1, [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
