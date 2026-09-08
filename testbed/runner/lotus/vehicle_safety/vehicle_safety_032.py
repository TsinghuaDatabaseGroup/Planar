#!/usr/bin/env python3
"""
vehicle_safety-032
Determine whether at least two distinct complaint-recall pairs match recent
Tesla Model 3 unintended-acceleration complaints to related recalls.
DAG: complaint FILTER -> SEM_FILTER; recall FILTER -> SEM_FILTER;
     JOIN(vehicle_id, component_id) -> GROUP_BY([]) -> PROJECT(Yes/No)
Output: label, metric: label_accuracy
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

TASK_ID = "vehicle_safety-032"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            [
                "complaint_id",
                "make",
                "model",
                "model_year",
                "vehicle_id",
                "component_id",
                "summary_text",
            ]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        selected_complaints = complaints[
            (complaints["make"] == "TESLA")
            & (complaints["model"] == "MODEL 3")
            & complaints["model_year"].isin([2020, 2021, 2022])
        ]
        tracker.record(
            "FILTER(make='TESLA' AND model='MODEL 3' AND model_year IN [2020,2021,2022])",
            len(complaints),
            len(selected_complaints),
        )

        with tracker.step(
            "SEM_FILTER(unintended acceleration or stuck accelerator pedal)",
            input_rows=len(selected_complaints),
        ) as step:
            acceleration_complaints = selected_complaints.sem_filter(
                "The complaint {summary_text} describes an unintended "
                "acceleration event or a stuck accelerator pedal."
            )
            step.set_output(acceleration_complaints)

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "recall_id",
                "vehicle_vehicle_id",
                "component_component_id",
                "vehicle_make",
                "vehicle_model",
                "vehicle_model_year",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "vehicle_id",
                "component_component_id": "component_id",
                "vehicle_make": "make",
                "vehicle_model": "model",
                "vehicle_model_year": "model_year",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("SCAN_DOCS(recalls)", None, len(recalls))

        selected_recalls = recalls[
            (recalls["make"] == "TESLA")
            & (recalls["model"] == "MODEL 3")
            & recalls["model_year"].isin([2020, 2021, 2022])
        ]
        tracker.record(
            "FILTER(vehicle.make='TESLA' AND vehicle.model='MODEL 3' AND vehicle.model_year IN [2020,2021,2022])",
            len(recalls),
            len(selected_recalls),
        )

        with tracker.step(
            "SEM_FILTER(accelerator-pedal or speed-control recall defect)",
            input_rows=len(selected_recalls),
        ) as step:
            accelerator_recalls = selected_recalls.sem_filter(
                "The recall defect summary {defect_summary} describes an "
                "accelerator-pedal or speed-control defect."
            )
            step.set_output(accelerator_recalls)

        joined = acceleration_complaints.merge(
            accelerator_recalls[["recall_id", "vehicle_id", "component_id"]],
            on=["vehicle_id", "component_id"],
            how="inner",
        )
        tracker.record(
            "JOIN(inner, on=[vehicle_id, component_id])",
            {
                "left": len(acceleration_complaints),
                "right": len(accelerator_recalls),
            },
            len(joined),
        )

        distinct_pairs = joined[["complaint_id", "recall_id"]].drop_duplicates()
        pair_count = len(distinct_pairs)
        tracker.record(
            "GROUP_BY([], n=count_distinct(complaint_id, recall_id))",
            len(joined),
            1,
        )

        answer = "Yes" if pair_count >= 2 else "No"
        tracker.record("PROJECT(label=(n>=2 ? 'Yes' : 'No'))", 1, 1)

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
