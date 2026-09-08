#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-016."""

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

TASK_ID = "vehicle_safety-016"
DATASET = "nhtsa_vehicle_safety"
EVENT_CONTEXTS = (
    "moving_or_in_traffic",
    "parked_idling_or_unclear",
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

        engine_complaints = complaints.loc[
            complaints["component_id"] == "ENGINE"
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), engine_complaints)

        relevance_plan = memory_dataset(TASK_ID, engine_complaints).sem_filter(
            filter=(
                "The complaint summary describes an engine stall, engine shutoff, "
                "loss of power, or failure to accelerate caused by engine operation."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        relevant = result_frame(relevance_result, engine_complaints)
        tracker.record_semantic(
            "sem_filter",
            len(engine_complaints),
            relevant,
            relevance_result,
            time.time() - started,
        )

        classification_plan = memory_dataset(
            f"{TASK_ID}-context", relevant
        ).sem_map(
            cols=[
                {
                    "name": "event_context",
                    "type": str,
                    "desc": (
                        "Exactly moving_or_in_traffic when the event happened "
                        "while moving, driving, or in traffic; otherwise exactly "
                        "parked_idling_or_unclear for parked, idling, starting, "
                        "or unclear situations."
                    ),
                }
            ],
            desc="Classify the context in which the reported event happened.",
            depends_on=["summary_text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            relevant,
            ["event_context"],
        )
        classified["event_context"] = classified["event_context"].map(
            lambda value: normalize_enum(value, EVENT_CONTEXTS)
            or "parked_idling_or_unclear"
        )
        tracker.record_semantic(
            "sem_map",
            len(relevant),
            classified,
            classification_result,
            time.time() - started,
        )

        total = len(classified)
        if total == 0:
            raise ValueError(f"{TASK_ID}: ratio denominator is zero")
        moving_count = int(
            classified["event_context"].eq("moving_or_in_traffic").sum()
        )
        tracker.record("groupby", len(classified), [total, moving_count])

        answer = round(moving_count / total, 3)
        tracker.record("project", 1, [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
