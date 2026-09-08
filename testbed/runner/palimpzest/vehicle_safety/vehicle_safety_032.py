#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-032."""

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
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-032"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            [
                "complaint_id",
                "make",
                "model",
                "model_year",
                "vehicle_id",
                "component_id",
                "summary_text",
            ],
        )
        tracker.record("scan", None, complaints)

        selected_complaints = complaints.loc[
            (complaints["make"] == "TESLA")
            & (complaints["model"] == "MODEL 3")
            & complaints["model_year"].isin([2020, 2021, 2022])
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), selected_complaints)

        complaint_plan = memory_dataset(
            f"{TASK_ID}-complaints", selected_complaints
        ).sem_filter(
            filter=(
                "The complaint summary describes an unintended acceleration "
                "event or a stuck accelerator pedal."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        complaint_result = complaint_plan.run(config)
        acceleration_complaints = result_frame(
            complaint_result,
            selected_complaints,
        )
        tracker.record_semantic(
            "sem_filter",
            len(selected_complaints),
            acceleration_complaints,
            complaint_result,
            time.time() - started,
        )

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "recall_id",
                "vehicle_vehicle_id",
                "component_component_id",
                "vehicle_make",
                "vehicle_model",
                "vehicle_model_year",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "vehicle_id",
                "component_component_id": "component_id",
                "vehicle_make": "make",
                "vehicle_model": "model",
                "vehicle_model_year": "model_year",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("scan", None, recalls)

        selected_recalls = recalls.loc[
            (recalls["make"] == "TESLA")
            & (recalls["model"] == "MODEL 3")
            & recalls["model_year"].isin([2020, 2021, 2022])
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), selected_recalls)

        recall_plan = memory_dataset(
            f"{TASK_ID}-recalls", selected_recalls
        ).sem_filter(
            filter=(
                "The recall defect summary describes an accelerator-pedal or "
                "speed-control defect."
            ),
            depends_on=["defect_summary"],
        )
        started = time.time()
        recall_result = recall_plan.run(config)
        accelerator_recalls = result_frame(recall_result, selected_recalls)
        tracker.record_semantic(
            "sem_filter",
            len(selected_recalls),
            accelerator_recalls,
            recall_result,
            time.time() - started,
        )

        joined = acceleration_complaints.merge(
            accelerator_recalls[["recall_id", "vehicle_id", "component_id"]],
            on=["vehicle_id", "component_id"],
            how="inner",
        )
        tracker.record(
            "join",
            {
                "left": len(acceleration_complaints),
                "right": len(accelerator_recalls),
            },
            joined,
        )

        pair_count = len(
            joined[["complaint_id", "recall_id"]].drop_duplicates()
        )
        tracker.record("groupby", len(joined), [pair_count])

        answer = "Yes" if pair_count >= 2 else "No"
        tracker.record("project", 1, [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
