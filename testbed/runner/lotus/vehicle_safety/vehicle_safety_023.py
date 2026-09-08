#!/usr/bin/env python3
"""
vehicle_safety-023
Return the five states with the most medically attended airbag deployment
hazard complaints, using state code as the tie-breaker.
DAG: SCAN_TABLE(complaints) -> FILTER(medical attention and state) ->
     SEM_FILTER(airbag-related) -> SEM_FILTER(deployment hazard) ->
     GROUP_BY(state) -> ORDER_BY(count DESC, state ASC) -> LIMIT(5)
Output: ordered list, metric: ordered_f1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, df_records, load_table, save_output, setup

TASK_ID = "vehicle_safety-023"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["state_of_incident", "medical_attention_flag", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        states = complaints["state_of_incident"]
        recorded = complaints[
            complaints["medical_attention_flag"]
            & states.notna()
            & (states.astype(str) != "")
        ]
        tracker.record(
            "FILTER(medical_attention_flag='true' AND state_of_incident<>'')",
            len(complaints),
            len(recorded),
        )

        with tracker.step(
            "SEM_FILTER(airbag-related complaint)", input_rows=len(recorded)
        ) as step:
            airbag_related = recorded.sem_filter(
                "The complaint {summary_text} is airbag-related, including airbags, "
                "inflators, pretensioners, or supplemental restraint system behavior."
            )
            step.set_output(airbag_related)

        with tracker.step(
            "SEM_FILTER(airbag deployment hazard)",
            input_rows=len(airbag_related),
        ) as step:
            deployment_hazards = airbag_related.sem_filter(
                "The complaint {summary_text} describes an airbag deployment "
                "failure, unintended deployment, inflator rupture, or a similar "
                "deployment hazard."
            )
            step.set_output(deployment_hazards)

        grouped = (
            deployment_hazards.groupby(
                "state_of_incident", as_index=False, dropna=False
            )
            .size()
            .rename(
                columns={
                    "state_of_incident": "state",
                    "size": "complaint_count",
                }
            )
        )
        tracker.record(
            "GROUP_BY([state_of_incident AS state], complaint_count=count(*))",
            len(deployment_hazards),
            len(grouped),
        )

        ordered = grouped.sort_values(
            ["complaint_count", "state"], ascending=[False, True]
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([complaint_count DESC, state ASC])",
            len(grouped),
            len(ordered),
        )

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("LIMIT(5)", len(ordered), len(top_five))
        answer = df_records(top_five[["state", "complaint_count"]])

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
