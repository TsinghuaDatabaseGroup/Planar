#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-031."""

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

TASK_ID = "vehicle_safety-031"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "vehicle_id", "component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        airbag_complaints = complaints.loc[
            complaints["component_id"] == "AIRBAG"
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), airbag_complaints)

        complaint_plan = memory_dataset(
            f"{TASK_ID}-complaints", airbag_complaints
        ).sem_filter(
            filter=(
                "The complaint summary describes airbag non-deployment, "
                "unintended deployment, inflator rupture, shrapnel, or another "
                "serious deployment hazard."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        complaint_result = complaint_plan.run(config)
        hazardous_complaints = result_frame(
            complaint_result,
            airbag_complaints,
        )
        tracker.record_semantic(
            "sem_filter",
            len(airbag_complaints),
            hazardous_complaints,
            complaint_result,
            time.time() - started,
        )

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_vehicle_id",
                "component_component_id",
                "remedy_corrective_action",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "vehicle_id",
                "component_component_id": "component_id",
                "remedy_corrective_action": "corrective_action",
            }
        )
        tracker.record("scan", None, recalls)

        airbag_recalls = recalls.loc[
            recalls["component_id"] == "AIRBAG"
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), airbag_recalls)

        recall_plan = memory_dataset(
            f"{TASK_ID}-recalls", airbag_recalls
        ).sem_filter(
            filter=(
                "The corrective action describes replacement of an airbag "
                "inflator, airbag module, or inflator assembly."
            ),
            depends_on=["corrective_action"],
        )
        started = time.time()
        recall_result = recall_plan.run(config)
        replacement_recalls = result_frame(recall_result, airbag_recalls)
        tracker.record_semantic(
            "sem_filter",
            len(airbag_recalls),
            replacement_recalls,
            recall_result,
            time.time() - started,
        )

        joined = hazardous_complaints.merge(
            replacement_recalls[["vehicle_id", "campaign_number"]],
            on="vehicle_id",
            how="inner",
        )
        tracker.record(
            "join",
            {
                "left": len(hazardous_complaints),
                "right": len(replacement_recalls),
            },
            joined,
        )

        answer = int(joined["campaign_number"].nunique())
        tracker.record("groupby", len(joined), [answer])
        tracker.record("project", 1, [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
