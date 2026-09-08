#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-033."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

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

TASK_ID = "vehicle_safety-033"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_vehicle_id",
                "component_component_id",
                "campaign_report_received_date",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "vehicle_id",
                "component_component_id": "component_id",
            }
        )
        tracker.record("scan", None, recalls)

        report_dates = recalls["campaign_report_received_date"]
        dated_recalls = recalls.loc[
            report_dates.notna()
            & report_dates.astype(str).str.strip().ne("")
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), dated_recalls)

        extraction_plan = memory_dataset(
            f"{TASK_ID}-recalls", dated_recalls
        ).sem_map(
            cols=[
                {
                    "name": "defect_root_cause",
                    "type": str,
                    "desc": (
                        "A short root-cause phrase of at most ten words extracted "
                        "from the recall defect summary."
                    ),
                }
            ],
            desc="Extract the root cause stated in the recall defect summary.",
            depends_on=["risk_defect_summary"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted_recalls = result_frame(
            extraction_result,
            dated_recalls,
            ["defect_root_cause"],
        )
        tracker.record_semantic(
            "sem_map",
            len(dated_recalls),
            extracted_recalls,
            extraction_result,
            time.time() - started,
        )

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

        extracted_recalls = extracted_recalls.copy()
        complaints = complaints.copy()
        extracted_recalls["_report_date"] = pd.to_datetime(
            extracted_recalls["campaign_report_received_date"],
            errors="coerce",
        )
        complaints["_complaint_date"] = pd.to_datetime(
            complaints["received_date"],
            errors="coerce",
        )
        key_matches = extracted_recalls.merge(
            complaints,
            on=["vehicle_id", "component_id"],
            how="inner",
        )
        key_matches["days_between"] = (
            key_matches["_report_date"] - key_matches["_complaint_date"]
        ).dt.days
        long_lag_pairs = key_matches.loc[
            (key_matches["_complaint_date"] < key_matches["_report_date"])
            & key_matches["days_between"].ge(1500)
        ].copy()
        long_lag_pairs["days_between"] = long_lag_pairs[
            "days_between"
        ].astype(int)
        long_lag_pairs = long_lag_pairs.drop(
            columns=["_report_date", "_complaint_date"]
        )
        long_lag_pairs = long_lag_pairs.reset_index(drop=True)
        tracker.record(
            "join",
            {"left": len(extracted_recalls), "right": len(complaints)},
            long_lag_pairs,
        )

        mechanism_plan = memory_dataset(
            f"{TASK_ID}-pairs", long_lag_pairs
        ).sem_filter(
            filter=(
                "The complaint summary describes the same defect mechanism as "
                "the recall defect summary."
            ),
            depends_on=["summary_text", "risk_defect_summary"],
        )
        started = time.time()
        mechanism_result = mechanism_plan.run(config)
        mechanism_matches = result_frame(mechanism_result, long_lag_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(long_lag_pairs),
            mechanism_matches,
            mechanism_result,
            time.time() - started,
        )

        result = mechanism_matches[
            [
                "complaint_id",
                "campaign_number",
                "days_between",
                "defect_root_cause",
            ]
        ].copy()
        tracker.record("project", len(mechanism_matches), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
