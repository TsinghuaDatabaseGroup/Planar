#!/usr/bin/env python3
"""
vehicle_safety-012
For Hyundai or Kia Park Outside recalls, count same-make, same-component
complaints describing fire, smoke, burning, melting, or electrical shorts.
DAG: complaint FILTER -> SEM_FILTER; recall FILTER -> DEDUP(campaign); then
     JOIN(make, component_id) -> GROUP_BY(campaign) -> PROJECT
Output: table, metric: table_f1
"""
import os
import sys

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

TASK_ID = "vehicle_safety-012"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        selected_makes = complaints[complaints["make"].isin(["KIA", "HYUNDAI"])]
        tracker.record(
            "FILTER(make IN ['KIA','HYUNDAI'])",
            len(complaints),
            len(selected_makes),
        )

        with tracker.step(
            "SEM_FILTER(fire, smoke, burning, melting, or electrical short)",
            input_rows=len(selected_makes),
        ) as step:
            hazardous_complaints = selected_makes.sem_filter(
                "The complaint {summary_text} describes fire, smoke, a burning "
                "smell, melting, or an electrical short related to the vehicle system."
            )
            step.set_output(hazardous_complaints)

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_park_outside",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_make": "make",
                "component_component_id": "component_id",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        park_outside = recalls[
            recalls["make"].isin(["KIA", "HYUNDAI"])
            & (recalls["risk_park_outside"] == "TRUE")
        ]
        tracker.record(
            "FILTER(make IN ['KIA','HYUNDAI'] AND risk.park_outside='TRUE')",
            len(recalls),
            len(park_outside),
        )

        deduped_recalls = park_outside.drop_duplicates(subset=["campaign_number"])
        tracker.record(
            "DEDUP([campaign.number])", len(park_outside), len(deduped_recalls)
        )

        joined = hazardous_complaints.merge(
            deduped_recalls[
                ["campaign_number", "make", "component_id", "defect_summary"]
            ],
            on=["make", "component_id"],
            how="inner",
        )
        tracker.record(
            "JOIN(inner, on=[make=r.make, component_id=r.component_id])",
            {
                "left": len(hazardous_complaints),
                "right": len(deduped_recalls),
            },
            len(joined),
        )

        grouped = joined.groupby(
            ["campaign_number", "make", "component_id"], as_index=False
        ).agg(
            matched_complaint_count=("complaint_id", "nunique"),
            defect_summary=("defect_summary", "first"),
        )
        tracker.record(
            "GROUP_BY([campaign.number, make, component_id], matched_complaint_count, defect_summary=any())",
            len(joined),
            len(grouped),
        )

        result = grouped[
            [
                "campaign_number",
                "make",
                "component_id",
                "matched_complaint_count",
                "defect_summary",
            ]
        ]
        tracker.record(
            "PROJECT([campaign_number, make, component_id, matched_complaint_count, defect_summary])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
