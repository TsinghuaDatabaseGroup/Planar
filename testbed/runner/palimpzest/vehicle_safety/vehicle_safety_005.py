#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-005."""

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
    load_jsonl,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-005"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        recalls = load_jsonl(DATASET, "recalls.jsonl")
        tracker.record("scan", None, recalls)

        filtered = recalls.loc[
            (recalls["vehicle_make"] == "TESLA")
            & (recalls["risk_park_outside"] == "FALSE")
            & (recalls["risk_do_not_drive"] == "FALSE")
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), filtered)

        semantic_plan = memory_dataset(TASK_ID, filtered).sem_filter(
            filter=(
                "The corrective action describes an over-the-air software "
                "update, and the defect summary describes a steering or "
                "motion-control issue. Both conditions must hold."
            ),
            depends_on=["remedy_corrective_action", "risk_defect_summary"],
        )
        started = time.time()
        semantic_result = semantic_plan.run(config)
        matching = result_frame(semantic_result, filtered)
        tracker.record_semantic(
            "sem_filter",
            len(filtered),
            matching,
            semantic_result,
            time.time() - started,
        )

        result = matching[
            [
                "campaign_number",
                "vehicle_model",
                "vehicle_model_year",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_model": "model",
                "vehicle_model_year": "model_year",
                "risk_defect_summary": "defect_summary",
            }
        )
        result = result.reset_index(drop=True)
        tracker.record("project", len(matching), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
