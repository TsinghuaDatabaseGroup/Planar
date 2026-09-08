#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for vehicle_safety-015."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_jsonl,
    load_table,
    memory_dataset,
    result_frame,
    run_optimized_plan,
    save_output,
)

TASK_ID = "vehicle_safety-015"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
DATASET = "nhtsa_vehicle_safety"
JOIN_COLUMNS = [
    "complaint_id",
    "summary_text",
    "recall_row_id",
    "campaign_number",
    "defect_summary",
]


def main() -> None:
    config = get_config(
        max_tokens=4096,
        all_optimizations=True,
        include_small_model=True,
    )
    tracker = StepTracker(TASK_ID, optimizer_strategy=OPTIMIZER)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)
        selected_complaints = complaints.loc[
            (complaints["make"] == "DODGE") & (complaints["component_id"] == "ELECTRICAL"),
            ["complaint_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), selected_complaints)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "recall_id",
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "recall_id": "recall_row_id",
                "vehicle_make": "make",
                "component_component_id": "component_id",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("scan", None, recalls)
        selected_recalls = recalls.loc[
            (recalls["make"] == "DODGE") & (recalls["component_id"] == "ELECTRICAL"),
            ["recall_row_id", "campaign_number", "defect_summary"],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), selected_recalls)

        plan = memory_dataset(
            f"{TASK_ID}-complaints",
            selected_complaints,
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", selected_recalls),
            condition=(
                "Keep a pair only when summary_text and defect_summary describe "
                "the same concrete electrical failure and the shared failure has "
                "a safety consequence such as stall, fire risk, loss of motive "
                "power, loss of lighting or warnings, unintended cruise "
                "behavior, or warning-system failure. Broad ELECTRICAL-category "
                "overlap is insufficient."
            ),
            depends_on=JOIN_COLUMNS,
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        matched = result_frame(semantic_result)
        if matched.empty:
            matched = pd.DataFrame(columns=JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(selected_complaints), "right": len(selected_recalls)},
            matched,
            semantic_result,
            time.time() - started,
        )

        distinct_pairs = matched.drop_duplicates(["complaint_id", "recall_row_id"]).reset_index(drop=True)
        grouped = (
            distinct_pairs.groupby("campaign_number", as_index=False)["recall_row_id"]
            .count()
            .rename(columns={"recall_row_id": "matched_pair_count"})
            .sort_values(
                ["matched_pair_count", "campaign_number"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(matched), grouped)
        tracker.record("order_by", len(grouped), grouped)
        projected = grouped[["campaign_number", "matched_pair_count"]]
        tracker.record("project", len(grouped), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
