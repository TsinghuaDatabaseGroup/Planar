#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-021."""

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
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-021"
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
                "vehicle_id",
                "make",
                "model",
                "component_id",
                "injury_count",
                "death_count",
                "summary_text",
            ],
        )
        tracker.record("scan", None, complaints)

        serious_airbag = complaints.loc[
            (complaints["make"] == "NISSAN")
            & (complaints["model"] == "ALTIMA")
            & (complaints["component_id"] == "AIRBAG")
            & (
                complaints["injury_count"].gt(0)
                | complaints["death_count"].gt(0)
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), serious_airbag)

        relevance_plan = memory_dataset(TASK_ID, serious_airbag).sem_filter(
            filter=(
                "The complaint summary describes an airbag inflator rupture, "
                "airbag explosion, shrapnel, or metal fragments causing injury "
                "or possible serious injury."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        rupture_complaints = result_frame(relevance_result, serious_airbag)
        tracker.record_semantic(
            "sem_filter",
            len(serious_airbag),
            rupture_complaints,
            relevance_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-failure", rupture_complaints
        ).sem_map(
            cols=[
                {
                    "name": "airbag_failure_mode",
                    "type": str,
                    "desc": "The primary airbag failure mode as a short canonical phrase.",
                }
            ],
            desc="Extract the primary airbag failure mode from the complaint.",
            depends_on=["summary_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            rupture_complaints,
            ["airbag_failure_mode"],
        )
        tracker.record_semantic(
            "sem_map",
            len(rupture_complaints),
            extracted,
            extraction_result,
            time.time() - started,
        )

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            ["campaign_number", "vehicle_vehicle_id", "component_component_id"]
        ].rename(columns={"vehicle_vehicle_id": "vehicle_id"})
        tracker.record("scan", None, recalls)

        airbag_recalls = recalls.loc[
            recalls["component_component_id"] == "AIRBAG"
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), airbag_recalls)

        recall_keys = airbag_recalls[
            ["vehicle_id", "campaign_number"]
        ].drop_duplicates().reset_index(drop=True)
        tracker.record("distinct", len(airbag_recalls), recall_keys)

        joined = extracted.merge(recall_keys, on="vehicle_id", how="inner")
        tracker.record(
            "join",
            {"left": len(extracted), "right": len(recall_keys)},
            joined,
        )

        result = joined[
            [
                "complaint_id",
                "vehicle_id",
                "campaign_number",
                "airbag_failure_mode",
            ]
        ].copy()
        tracker.record("project", len(joined), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
