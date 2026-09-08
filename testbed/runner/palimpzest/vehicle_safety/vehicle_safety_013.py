#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for vehicle_safety-013."""

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

TASK_ID = "vehicle_safety-013"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
DATASET = "nhtsa_vehicle_safety"
JOIN_COLUMNS = [
    "complaint_id",
    "vehicle_id",
    "component_id",
    "received_date",
    "summary_text",
    "recall_id",
    "campaign_number",
    "recall_vehicle_id",
    "recall_component_id",
    "report_received_date",
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
            [
                "complaint_id",
                "vehicle_id",
                "component_id",
                "received_date",
                "summary_text",
            ],
        )
        tracker.record("scan", None, complaints)
        dated_complaints = complaints.loc[
            complaints["received_date"].notna() & complaints["received_date"].astype(str).str.strip().ne("")
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), dated_complaints)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "recall_id",
                "campaign_number",
                "vehicle_vehicle_id",
                "component_component_id",
                "campaign_report_received_date",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "recall_vehicle_id",
                "component_component_id": "recall_component_id",
                "campaign_report_received_date": "report_received_date",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("scan", None, recalls)
        dated_recalls = recalls.loc[
            recalls["report_received_date"].notna() & recalls["report_received_date"].astype(str).str.strip().ne("")
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), dated_recalls)

        plan = memory_dataset(
            f"{TASK_ID}-complaints",
            dated_complaints,
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", dated_recalls),
            condition=(
                "Evaluate only pairs where vehicle_id equals recall_vehicle_id "
                "and component_id equals recall_component_id, and where "
                "report_received_date is at least 90 days after received_date. "
                "Keep a pair only when summary_text and defect_summary describe "
                "the same concrete underlying defect mechanism, not merely the "
                "same broad component or symptom."
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
            {"left": len(dated_complaints), "right": len(dated_recalls)},
            matched,
            semantic_result,
            time.time() - started,
        )

        distinct_pairs = matched.drop_duplicates(["complaint_id", "recall_id"]).reset_index(drop=True)
        grouped = pd.DataFrame.from_records([{"matched_pair_count": len(distinct_pairs)}])
        tracker.record("groupby", len(matched), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
