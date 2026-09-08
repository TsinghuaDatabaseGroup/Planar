#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-030."""

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

TASK_ID = "vehicle_safety-030"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "model_year", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        recent_complaints = complaints.loc[
            complaints["model_year"].isin([2023, 2024, 2025])
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), recent_complaints)

        complaint_plan = memory_dataset(
            f"{TASK_ID}-complaints", recent_complaints
        ).sem_filter(
            filter=(
                "The complaint summary describes an EV or hybrid component issue "
                "involving a battery, charging system, battery management system "
                "(BMS), electric drive, hybrid drivetrain, high-voltage system, or "
                "powertrain-control software."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        complaint_result = complaint_plan.run(config)
        ev_hybrid_complaints = result_frame(
            complaint_result,
            recent_complaints,
        )
        tracker.record_semantic(
            "sem_filter",
            len(recent_complaints),
            ev_hybrid_complaints,
            complaint_result,
            time.time() - started,
        )

        complaint_counts = (
            ev_hybrid_complaints.groupby(
                "make",
                as_index=False,
                dropna=False,
            )
            .agg(ev_hybrid_complaint_count=("complaint_id", "nunique"))
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(ev_hybrid_complaints), complaint_counts)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "campaign_report_received_date",
                "risk_defect_summary",
                "remedy_corrective_action",
            ]
        ].rename(
            columns={
                "vehicle_make": "make",
                "risk_defect_summary": "defect_summary",
                "remedy_corrective_action": "corrective_action",
            }
        )
        tracker.record("scan", None, recalls)

        report_dates = pd.to_datetime(
            recalls["campaign_report_received_date"],
            errors="coerce",
        )
        recent_recalls = recalls.loc[
            report_dates.between(
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2025-12-31"),
                inclusive="both",
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), recent_recalls)

        recall_plan = memory_dataset(
            f"{TASK_ID}-recalls", recent_recalls
        ).sem_filter(
            filter=(
                "The recall defect summary or corrective action discusses EV "
                "batteries, charging, high-voltage systems, electric drive, hybrid "
                "drivetrain, or powertrain-control software."
            ),
            depends_on=["defect_summary", "corrective_action"],
        )
        started = time.time()
        recall_result = recall_plan.run(config)
        ev_hybrid_recalls = result_frame(recall_result, recent_recalls)
        tracker.record_semantic(
            "sem_filter",
            len(recent_recalls),
            ev_hybrid_recalls,
            recall_result,
            time.time() - started,
        )

        recall_counts = (
            ev_hybrid_recalls.groupby("make", as_index=False, dropna=False)
            .agg(
                related_ev_hybrid_recall_count=(
                    "campaign_number",
                    "nunique",
                )
            )
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(ev_hybrid_recalls), recall_counts)

        joined = complaint_counts.merge(recall_counts, on="make", how="inner")
        tracker.record(
            "join",
            {"left": len(complaint_counts), "right": len(recall_counts)},
            joined,
        )

        ordered = joined.sort_values(
            ["ev_hybrid_complaint_count", "make"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(joined), ordered)

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_five)

        top_five.insert(0, "rank", range(1, len(top_five) + 1))
        result = top_five[
            [
                "rank",
                "make",
                "ev_hybrid_complaint_count",
                "related_ev_hybrid_recall_count",
            ]
        ].copy()
        tracker.record("project", len(top_five), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
