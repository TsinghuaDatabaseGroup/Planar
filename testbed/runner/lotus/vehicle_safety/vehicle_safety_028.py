#!/usr/bin/env python3
"""
vehicle_safety-028
Compare severe in-motion Toyota and Honda complaints for vehicles covered by
a same-vehicle recall describing a safety-critical defect.
DAG: complaint FILTER -> SEM_FILTER; recall FILTER -> SEM_FILTER -> DEDUP;
     JOIN(vehicle_id) -> GROUP_BY(make) -> PROJECT(comparison dictionary)
Output: dictionary, metric: f1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (
    StepTracker,
    Timer,
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-028"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "vehicle_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        selected_complaints = complaints[
            complaints["make"].isin(["TOYOTA", "HONDA"])
        ]
        tracker.record(
            "FILTER(make IN ['TOYOTA','HONDA'])",
            len(complaints),
            len(selected_complaints),
        )

        with tracker.step(
            "SEM_FILTER(severe safety event while vehicle was being driven)",
            input_rows=len(selected_complaints),
        ) as step:
            severe_complaints = selected_complaints.sem_filter(
                "The complaint {summary_text} describes a severe safety event "
                "that happened while the vehicle was being driven, including a "
                "crash, injury, death, fire, airbag non-deployment, loss of "
                "control, brake failure, loss of power, or towing after a "
                "driving incident."
            )
            step.set_output(severe_complaints)

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_vehicle_id",
                "vehicle_make",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "vehicle_id",
                "vehicle_make": "make",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        selected_recalls = recalls[recalls["make"].isin(["TOYOTA", "HONDA"])]
        tracker.record(
            "FILTER(vehicle.make IN ['TOYOTA','HONDA'])",
            len(recalls),
            len(selected_recalls),
        )

        with tracker.step(
            "SEM_FILTER(safety-critical recall defect)",
            input_rows=len(selected_recalls),
        ) as step:
            critical_recalls = selected_recalls.sem_filter(
                "The recall defect summary {defect_summary} describes a "
                "safety-critical defect that can cause a crash, injury, fire, "
                "loss of control, airbag failure, braking failure, or loss of "
                "motive power."
            )
            step.set_output(critical_recalls)

        deduped_recalls = critical_recalls.drop_duplicates(
            subset=["campaign_number", "vehicle_id"]
        )
        tracker.record(
            "DEDUP([campaign.number, vehicle_id])",
            len(critical_recalls),
            len(deduped_recalls),
        )

        joined = severe_complaints.merge(
            deduped_recalls[["vehicle_id"]],
            on="vehicle_id",
            how="inner",
        )
        tracker.record(
            "JOIN(inner, on=[vehicle_id=r.vehicle_id])",
            {
                "left": len(severe_complaints),
                "right": len(deduped_recalls),
            },
            len(joined),
        )

        grouped = joined.groupby("make", as_index=False).agg(
            matched_complaint_count=("complaint_id", "nunique")
        )
        tracker.record(
            "GROUP_BY([make], matched_complaint_count=count_distinct(complaint_id))",
            len(joined),
            len(grouped),
        )

        counts = dict(
            zip(grouped["make"], grouped["matched_complaint_count"], strict=True)
        )
        toyota_count = int(counts.get("TOYOTA", 0))
        honda_count = int(counts.get("HONDA", 0))
        if toyota_count > honda_count:
            higher_make = "TOYOTA"
        elif honda_count > toyota_count:
            higher_make = "HONDA"
        else:
            higher_make = "tie"

        answer = {
            "toyota_matched_complaint_count": toyota_count,
            "honda_matched_complaint_count": honda_count,
            "higher_matched_complaint_make": higher_make,
        }
        tracker.record(
            "PROJECT(format_as_dictionary, compare_counts)",
            len(grouped),
            1,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
