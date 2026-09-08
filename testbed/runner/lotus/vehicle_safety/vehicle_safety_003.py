#!/usr/bin/env python3
"""
vehicle_safety-003
For Dodge airbag complaints involving at least one injury and no fire, what
primary airbag failure mode is described in each case? For every complaint ID,
the model year and a short canonical failure-mode phrase.
DAG: SCAN_TABLE(complaints) -> FILTER(make='DODGE' AND component_id='AIRBAG'
     AND injury_count>=1 AND NOT fire_flag) -> SEM_EXTRACT(airbag_failure_mode)
     -> PROJECT([complaint_id, model_year, airbag_failure_mode])
Output: table, metric: table_f1
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_table, save_output, df_records, StepTracker, Timer

TASK_ID = "vehicle_safety-003"

def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as t:
        df = load_table("nhtsa_vehicle_safety", "complaints.csv")
        tracker.record("SCAN_TABLE(complaints)", None, len(df))

        filtered = df[
            (df["make"] == "DODGE")
            & (df["component_id"] == "AIRBAG")
            & (df["injury_count"] >= 1)
            & ~df["fire_flag"]
        ]
        tracker.record(
            "FILTER(make='DODGE' AND component_id='AIRBAG' AND injury_count>=1 AND NOT fire_flag)",
            len(df), len(filtered),
        )

        with tracker.step("SEM_EXTRACT(airbag_failure_mode)", input_rows=len(filtered)) as st:
            extracted = filtered.sem_extract(
                input_cols=["summary_text"],
                output_cols={
                    "airbag_failure_mode": (
                        "the primary airbag failure mode described in the complaint, "
                        "as a short canonical phrase, such as non-deployment, "
                        "unintended deployment, inflator rupture, occupant burn or "
                        "laceration, warning-light fault, or other"
                    )
                },
            )
            st.set_output(extracted)

        result = extracted[["complaint_id", "model_year", "airbag_failure_mode"]]
        tracker.record("PROJECT([complaint_id, model_year, airbag_failure_mode])",
                       len(extracted), len(result))
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
