#!/usr/bin/env python3
"""
vehicle_safety-030
Rank the five makes with the most recent-model-year EV or hybrid complaints
among makes that also have a recent related recall.
DAG: complaint FILTER -> SEM_FILTER -> GROUP_BY(make); recall FILTER ->
     SEM_FILTER -> GROUP_BY(make); JOIN(make) -> ORDER_BY -> LIMIT -> PROJECT
Output: ordered list, metric: ordered_f1
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (
    StepTracker,
    Timer,
    df_records,
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-030"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "model_year", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        recent_complaints = complaints[
            complaints["model_year"].isin([2023, 2024, 2025])
        ]
        tracker.record(
            "FILTER(model_year IN [2023,2024,2025])",
            len(complaints),
            len(recent_complaints),
        )

        with tracker.step(
            "SEM_FILTER(EV or hybrid component issue)",
            input_rows=len(recent_complaints),
        ) as step:
            ev_hybrid_complaints = recent_complaints.sem_filter(
                "The complaint {summary_text} describes an EV or hybrid "
                "component issue involving a battery, charging system, battery "
                "management system (BMS), electric drive, hybrid drivetrain, "
                "high-voltage system, or powertrain-control software."
            )
            step.set_output(ev_hybrid_complaints)

        complaint_counts = ev_hybrid_complaints.groupby("make", as_index=False).agg(
            ev_hybrid_complaint_count=("complaint_id", "nunique")
        )
        tracker.record(
            "GROUP_BY([make], ev_hybrid_complaint_count=count_distinct(complaint_id))",
            len(ev_hybrid_complaints),
            len(complaint_counts),
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
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
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        report_dates = pd.to_datetime(
            recalls["campaign_report_received_date"], errors="coerce"
        )
        recent_recalls = recalls[
            (report_dates >= pd.Timestamp("2023-01-01"))
            & (report_dates <= pd.Timestamp("2025-12-31"))
        ]
        tracker.record(
            "FILTER(campaign.report_received_date>='2023-01-01' AND campaign.report_received_date<='2025-12-31')",
            len(recalls),
            len(recent_recalls),
        )

        with tracker.step(
            "SEM_FILTER(recall discusses EV or hybrid systems)",
            input_rows=len(recent_recalls),
        ) as step:
            ev_hybrid_recalls = recent_recalls.sem_filter(
                "The recall defect summary {defect_summary} or corrective action "
                "{corrective_action} discusses EV batteries, charging, "
                "high-voltage systems, electric drive, hybrid drivetrain, or "
                "powertrain-control software."
            )
            step.set_output(ev_hybrid_recalls)

        recall_counts = ev_hybrid_recalls.groupby("make", as_index=False).agg(
            related_ev_hybrid_recall_count=("campaign_number", "nunique")
        )
        tracker.record(
            "GROUP_BY([make], related_ev_hybrid_recall_count=count_distinct(campaign.number))",
            len(ev_hybrid_recalls),
            len(recall_counts),
        )

        joined = complaint_counts.merge(recall_counts, on="make", how="inner")
        tracker.record(
            "JOIN(inner, on=[make=r.make])",
            {"left": len(complaint_counts), "right": len(recall_counts)},
            len(joined),
        )

        ordered = joined.sort_values(
            ["ev_hybrid_complaint_count", "make"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([ev_hybrid_complaint_count DESC, make ASC])",
            len(joined),
            len(ordered),
        )

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("LIMIT(5)", len(ordered), len(top_five))

        result = top_five.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result = result[
            [
                "rank",
                "make",
                "ev_hybrid_complaint_count",
                "related_ev_hybrid_recall_count",
            ]
        ]
        tracker.record(
            "PROJECT([rank=row_number(), make, ev_hybrid_complaint_count, related_ev_hybrid_recall_count])",
            len(top_five),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
