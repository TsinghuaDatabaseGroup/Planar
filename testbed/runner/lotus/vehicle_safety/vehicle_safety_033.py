#!/usr/bin/env python3
"""
vehicle_safety-033
Return long-lag complaint-recall pairs for the same vehicle and component when
their summaries describe the same defect mechanism.
DAG: recall FILTER -> SEM_EXTRACT -> JOIN(complaints, keys and >=1500 days) ->
     SEM_FILTER(same mechanism) -> PROJECT
Output: table, metric: table_f1
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

TASK_ID = "vehicle_safety-033"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
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
        tracker.record("SCAN_DOCS(recalls)", None, len(recalls))

        report_date_present = recalls[
            recalls["campaign_report_received_date"].notna()
            & recalls["campaign_report_received_date"].astype(str).str.strip().ne("")
        ].copy()
        tracker.record(
            "FILTER(campaign.report_received_date IS NOT NULL AND campaign.report_received_date<>'')",
            len(recalls),
            len(report_date_present),
        )

        with tracker.step(
            "SEM_EXTRACT(defect_root_cause<=10 words)",
            input_rows=len(report_date_present),
        ) as step:
            if report_date_present.empty:
                extracted_recalls = report_date_present.copy()
                extracted_recalls["defect_root_cause"] = pd.Series(dtype="object")
            else:
                extracted_recalls = report_date_present.sem_extract(
                    input_cols=["risk_defect_summary"],
                    output_cols={
                        "defect_root_cause": (
                            "A short root-cause phrase of at most 10 words extracted "
                            "from the recall defect summary."
                        )
                    },
                )
            step.set_output(extracted_recalls)

        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            [
                "complaint_id",
                "vehicle_id",
                "component_id",
                "received_date",
                "summary_text",
            ]
        ].copy()
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        extracted_recalls["_report_date"] = pd.to_datetime(
            extracted_recalls["campaign_report_received_date"], errors="coerce"
        )
        complaints["_complaint_date"] = pd.to_datetime(
            complaints["received_date"], errors="coerce"
        )

        key_matches = extracted_recalls.merge(
            complaints,
            on=["vehicle_id", "component_id"],
            how="inner",
        )
        day_differences = (
            key_matches["_report_date"] - key_matches["_complaint_date"]
        ).dt.days
        long_lag_pairs = key_matches[
            (key_matches["_complaint_date"] < key_matches["_report_date"])
            & (day_differences >= 1500)
        ].copy()
        long_lag_pairs["days_between"] = day_differences.loc[
            long_lag_pairs.index
        ].astype(int)
        tracker.record(
            "JOIN(inner, on=[vehicle_id, component_id, received_date<report_received_date, date_diff>=1500])",
            {
                "left": len(extracted_recalls),
                "right": len(complaints),
            },
            len(long_lag_pairs),
        )

        with tracker.step(
            "SEM_FILTER(complaint and recall describe same defect mechanism)",
            input_rows=len(long_lag_pairs),
        ) as step:
            if long_lag_pairs.empty:
                mechanism_matches = long_lag_pairs.copy()
            else:
                mechanism_matches = long_lag_pairs.sem_filter(
                    "The complaint summary {summary_text} describes the same defect "
                    "mechanism as the recall defect summary {risk_defect_summary}."
                )
            step.set_output(mechanism_matches)

        result = mechanism_matches[
            [
                "complaint_id",
                "campaign_number",
                "days_between",
                "defect_root_cause",
            ]
        ]
        tracker.record(
            "PROJECT([complaint_id, campaign_number, days_between, defect_root_cause])",
            len(mechanism_matches),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
