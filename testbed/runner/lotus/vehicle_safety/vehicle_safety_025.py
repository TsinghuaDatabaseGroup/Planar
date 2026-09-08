#!/usr/bin/env python3
"""
vehicle_safety-025
Rank the four makes with the most phantom-braking ADAS complaints among makes
that also have matching automatic-braking or speed-control ADAS recalls.
DAG: complaint FILTER -> SEM_FILTER -> GROUP_BY; recall FILTER -> SEM_FILTER ->
     GROUP_BY; then JOIN(make) -> ORDER_BY -> LIMIT(4) -> PROJECT(rank)
Output: ordered list, metric: ordered_f1
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

TASK_ID = "vehicle_safety-025"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        adas_complaints = complaints[complaints["component_id"] == "ADAS"]
        tracker.record(
            "FILTER(component_id='ADAS')",
            len(complaints),
            len(adas_complaints),
        )

        with tracker.step(
            "SEM_FILTER(phantom braking)", input_rows=len(adas_complaints)
        ) as step:
            phantom_braking = adas_complaints.sem_filter(
                "The complaint {summary_text} specifically describes phantom "
                "braking: sudden, automatic, unprovoked braking by a driver-assist "
                "system."
            )
            step.set_output(phantom_braking)

        complaint_counts = phantom_braking.groupby("make", as_index=False).agg(
            phantom_braking_complaint_count=("complaint_id", "nunique")
        )
        tracker.record(
            "GROUP_BY([make], phantom_braking_complaint_count=count_distinct(complaint_id))",
            len(phantom_braking),
            len(complaint_counts),
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(columns={"vehicle_make": "make"})
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        adas_recalls = recalls[recalls["component_component_id"] == "ADAS"]
        tracker.record(
            "FILTER(component.component_id='ADAS')",
            len(recalls),
            len(adas_recalls),
        )

        with tracker.step(
            "SEM_FILTER(automatic braking or speed-control recall defect)",
            input_rows=len(adas_recalls),
        ) as step:
            matching_recalls = adas_recalls.sem_filter(
                "The recall defect summary {risk_defect_summary} discusses automatic "
                "braking, driver-assist braking, adaptive cruise braking, or "
                "speed-control behavior as the recall defect."
            )
            step.set_output(matching_recalls)

        recall_counts = matching_recalls.groupby("make", as_index=False).agg(
            matching_adas_recall_count=("campaign_number", "nunique")
        )
        tracker.record(
            "GROUP_BY([make], matching_adas_recall_count=count_distinct(campaign.number))",
            len(matching_recalls),
            len(recall_counts),
        )

        joined = complaint_counts.merge(recall_counts, on="make", how="inner")
        tracker.record(
            "JOIN(inner, on=[make=r.make])",
            {"left": len(complaint_counts), "right": len(recall_counts)},
            len(joined),
        )

        ordered = joined.sort_values(
            ["phantom_braking_complaint_count", "make"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([phantom_braking_complaint_count DESC, make ASC])",
            len(joined),
            len(ordered),
        )

        top_four = ordered.head(4).copy()
        tracker.record("LIMIT(4)", len(ordered), len(top_four))

        top_four.insert(0, "rank", range(1, len(top_four) + 1))
        result = top_four[
            [
                "rank",
                "make",
                "phantom_braking_complaint_count",
                "matching_adas_recall_count",
            ]
        ]
        tracker.record(
            "PROJECT([rank=row_number(), make, phantom_braking_complaint_count, matching_adas_recall_count])",
            len(top_four),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
