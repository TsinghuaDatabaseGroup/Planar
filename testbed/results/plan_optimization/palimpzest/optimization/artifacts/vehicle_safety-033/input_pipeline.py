#!/usr/bin/env python3
"""Plan-optimization pipeline for vehicle_safety-033."""

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
    run_plan_optimization,
    save_output,
)

TASK_ID = "vehicle_safety-033"
DATASET = "nhtsa_vehicle_safety"


def add_days_between(record: dict) -> dict:
    recall_date = pd.to_datetime(
        record.get("campaign_report_received_date"), errors="coerce"
    )
    complaint_date = pd.to_datetime(record.get("received_date"), errors="coerce")
    if pd.isna(recall_date) or pd.isna(complaint_date):
        return {"days_between": None}
    return {"days_between": int((recall_date - complaint_date).days)}


def is_long_lag(record: dict) -> bool:
    try:
        return int(record.get("days_between")) >= 1_500
    except (TypeError, ValueError):
        return False


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        recalls = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "vehicle_vehicle_id",
            "component_component_id",
            "campaign_report_received_date",
            "risk_defect_summary",
        ]].rename(
            columns={
                "vehicle_vehicle_id": "vehicle_id",
                "component_component_id": "component_id",
            }
        )
        report_dates = recalls["campaign_report_received_date"]
        dated_recalls = recalls.loc[
            report_dates.notna() & report_dates.astype(str).str.strip().ne("")
        ].reset_index(drop=True)
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

        recall_plan = memory_dataset(f"{TASK_ID}-recalls", dated_recalls).sem_map(
            cols=[
                {
                    "name": "defect_root_cause",
                    "type": str,
                    "desc": (
                        "A root-cause phrase of no more than ten words from the "
                        "recall defect summary."
                    ),
                }
            ],
            desc="Extract the root cause stated in the recall defect summary.",
            depends_on=["risk_defect_summary"],
        )
        pairs = recall_plan.join(
            memory_dataset(f"{TASK_ID}-complaints", complaints),
            on=["vehicle_id", "component_id"],
            how="inner",
        )
        pairs = pairs.map(
            add_days_between,
            cols=[
                {
                    "name": "days_between",
                    "type": int | None,
                    "desc": "Days from complaint receipt to recall report.",
                }
            ],
            depends_on=["campaign_report_received_date", "received_date"],
        )
        pairs = pairs.filter(is_long_lag, depends_on=["days_between"])
        plan = pairs.sem_filter(
            (
                "Keep this complaint-recall pair only if the complaint summary "
                "describes the same defect mechanism as the recall defect "
                "summary."
            ),
            depends_on=["summary_text", "risk_defect_summary"],
        )
        plan = plan.project(
            ["complaint_id", "campaign_number", "days_between", "defect_root_cause"]
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True)
        tracker.record_semantic(
            "optimized_plan",
            {"recalls": len(dated_recalls), "complaints": len(complaints)},
            output,
            optimized.result,
            time.time() - started,
        )
        answer = df_records(
            output[
                [
                    "complaint_id",
                    "campaign_number",
                    "days_between",
                    "defect_root_cause",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
