#!/usr/bin/env python3
"""
vehicle_safety-031
Count distinct airbag recall campaigns whose replacement remedy has at least
one same-vehicle complaint describing a serious airbag deployment hazard.
DAG: complaint FILTER -> SEM_FILTER; recall FILTER -> SEM_FILTER;
     JOIN(vehicle_id) -> GROUP_BY([]) -> PROJECT(n)
Output: scalar, metric: exact_match
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

TASK_ID = "vehicle_safety-031"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "vehicle_id", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        airbag_complaints = complaints[complaints["component_id"] == "AIRBAG"]
        tracker.record(
            "FILTER(component_id='AIRBAG')",
            len(complaints),
            len(airbag_complaints),
        )

        with tracker.step(
            "SEM_FILTER(serious airbag deployment hazard)",
            input_rows=len(airbag_complaints),
        ) as step:
            hazardous_complaints = airbag_complaints.sem_filter(
                "The complaint {summary_text} describes airbag non-deployment, "
                "unintended deployment, inflator rupture, shrapnel, or another "
                "serious deployment hazard."
            )
            step.set_output(hazardous_complaints)

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_vehicle_id",
                "component_component_id",
                "remedy_corrective_action",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "vehicle_id",
                "component_component_id": "component_id",
                "remedy_corrective_action": "corrective_action",
            }
        )
        tracker.record("SCAN_DOCS(recalls)", None, len(recalls))

        airbag_recalls = recalls[recalls["component_id"] == "AIRBAG"]
        tracker.record(
            "FILTER(component.component_id='AIRBAG')",
            len(recalls),
            len(airbag_recalls),
        )

        with tracker.step(
            "SEM_FILTER(replacement of airbag inflator or module)",
            input_rows=len(airbag_recalls),
        ) as step:
            replacement_recalls = airbag_recalls.sem_filter(
                "The corrective action {corrective_action} describes replacement "
                "of an airbag inflator, airbag module, or inflator assembly."
            )
            step.set_output(replacement_recalls)

        joined = hazardous_complaints.merge(
            replacement_recalls[["vehicle_id", "campaign_number"]],
            on="vehicle_id",
            how="inner",
        )
        tracker.record(
            "JOIN(inner, on=[vehicle_id])",
            {
                "left": len(hazardous_complaints),
                "right": len(replacement_recalls),
            },
            len(joined),
        )

        campaign_count = int(joined["campaign_number"].nunique())
        tracker.record(
            "GROUP_BY([], n=count_distinct(campaign.number))",
            len(joined),
            1,
        )

        answer = campaign_count
        tracker.record("PROJECT([n])", 1, 1)

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
