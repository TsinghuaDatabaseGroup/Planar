#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-002."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_jsonl,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-002"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        recalls = load_jsonl(DATASET, "recalls.jsonl")
        tracker.record("scan", None, recalls)

        seatbelt_recalls = recalls.loc[
            recalls["component_component_id"] == "SEATBELT",
            ["risk_defect_summary"],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), seatbelt_recalls)

        aggregate_plan = memory_dataset(TASK_ID, seatbelt_recalls).sem_agg(
            col={
                "name": "answer",
                "type": str,
                "desc": "Exactly Yes or No.",
            },
            agg=(
                "Determine whether every input defect summary describes a defect "
                "that could prevent the belt from restraining an occupant during "
                "a crash, and none describes a purely cosmetic or comfort-only "
                "issue. Return Yes only when both universal conditions hold; "
                "otherwise return No."
            ),
            depends_on=["risk_defect_summary"],
        )
        started = time.time()
        aggregate_result = aggregate_plan.run(config)
        aggregate_frame = result_frame(aggregate_result)
        tracker.record_semantic(
            "sem_agg",
            len(seatbelt_recalls),
            aggregate_frame,
            aggregate_result,
            time.time() - started,
        )

        if aggregate_frame.empty:
            raise ValueError(f"{TASK_ID}: semantic aggregate returned no answer")
        normalized = normalize_enum(
            aggregate_frame.iloc[0]["answer"],
            ("yes", "no"),
        )
        if normalized is None:
            raise ValueError(f"{TASK_ID}: semantic aggregate returned invalid label")
        answer = normalized.capitalize()

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
