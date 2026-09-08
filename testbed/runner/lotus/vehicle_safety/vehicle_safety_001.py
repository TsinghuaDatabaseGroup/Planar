#!/usr/bin/env python3
"""
vehicle_safety-001
Among 2022 model-year Tesla Model 3 complaints that reported a crash but no
fire, how many describe phantom braking or sudden unintended acceleration?
DAG: SCAN_TABLE(complaints) -> FILTER(vehicle_id='TESLA__MODEL_3__2022' AND
     crash_flag AND NOT fire_flag) -> SEM_FILTER(phantom braking or SUA)
     -> GROUP_BY([], count(*))
Output: scalar (integer), metric: exact_match
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_table, save_output, StepTracker, Timer

TASK_ID = "vehicle_safety-001"

def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as t:
        df = load_table("nhtsa_vehicle_safety", "complaints.csv")
        tracker.record("SCAN_TABLE(complaints)", None, len(df))

        filtered = df[
            (df["vehicle_id"] == "TESLA__MODEL_3__2022")
            & df["crash_flag"]
            & ~df["fire_flag"]
        ]
        tracker.record(
            "FILTER(vehicle_id='TESLA__MODEL_3__2022' AND crash_flag AND NOT fire_flag)",
            len(df), len(filtered),
        )

        with tracker.step("SEM_FILTER(phantom braking or sudden unintended acceleration)",
                          input_rows=len(filtered)) as st:
            hits = filtered.sem_filter(
                "The complaint {summary_text} describes phantom braking or "
                "sudden unintended acceleration"
            )
            st.set_output(hits)

        answer = int(len(hits))
        tracker.record("GROUP_BY([], count(*))", len(hits), 1)

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
