#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-024."""

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
    load_table,
    memory_dataset,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "vehicle_safety-024"
DATASET = "nhtsa_vehicle_safety"


def normalize_failure_mode(value) -> str:
    normalized = " ".join(str(value).strip().removesuffix(".").lower().split())
    if normalized in {"", "none", "null", "nan", "<na>"}:
        return "unspecified"
    return normalized


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            [
                "model_year",
                "component_id",
                "vehicle_towed_flag",
                "injury_count",
                "summary_text",
            ],
        )
        tracker.record("scan", None, complaints)

        recent = complaints.loc[
            complaints["model_year"].isin([2024, 2025])
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), recent)

        relevance_plan = memory_dataset(TASK_ID, recent).sem_filter(
            filter=(
                "The complaint summary describes a tangible safety event, not "
                "only a paperwork issue, recall notice, cosmetic concern, or "
                "routine service request."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        tangible = result_frame(relevance_result, recent)
        tracker.record_semantic(
            "sem_filter",
            len(recent),
            tangible,
            relevance_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-failure", tangible
        ).sem_map(
            cols=[
                {
                    "name": "failure_mode",
                    "type": str,
                    "desc": (
                        "A short canonical lower-case failure-mode label for the "
                        "primary failure described in the complaint."
                    ),
                }
            ],
            desc="Extract the primary failure mode from the complaint summary.",
            depends_on=["summary_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            tangible,
            ["failure_mode"],
        )
        extracted["failure_mode"] = extracted["failure_mode"].map(
            normalize_failure_mode
        )
        tracker.record_semantic(
            "sem_map",
            len(tangible),
            extracted,
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("component_id", as_index=False, dropna=False)
            .agg(
                complaint_count=("component_id", "size"),
                towed_rate=("vehicle_towed_flag", "mean"),
                avg_injury=("injury_count", "mean"),
                dominant_failure_mode=("failure_mode", stable_mode),
            )
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(extracted), grouped)

        result = grouped[
            [
                "component_id",
                "complaint_count",
                "towed_rate",
                "avg_injury",
                "dominant_failure_mode",
            ]
        ].copy()
        tracker.record("project", len(grouped), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
