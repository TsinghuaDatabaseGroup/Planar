#!/usr/bin/env python3
"""
vehicle_safety-007
Among Hyundai Elantra complaints where at least one occupant was injured but
no fire was reported, how many describe a loss of vehicle control while driving?
DAG: SCAN_TABLE(complaints) -> FILTER(make='HYUNDAI' AND model='ELANTRA' AND
     injury_count>=1 AND NOT fire_flag) -> SEM_FILTER(loss of vehicle control
     while driving) -> GROUP_BY([], count(*))
Output: scalar (integer), metric: exact_match
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_table, save_output, StepTracker, Timer

TASK_ID = "vehicle_safety-007"

def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as t:
        df = load_table("nhtsa_vehicle_safety", "complaints.csv")
        tracker.record("SCAN_TABLE(complaints)", None, len(df))

        filtered = df[
            (df["make"] == "HYUNDAI")
            & (df["model"] == "ELANTRA")
            & (df["injury_count"] >= 1)
            & ~df["fire_flag"]
        ]
        tracker.record(
            "FILTER(make='HYUNDAI' AND model='ELANTRA' AND injury_count>=1 AND NOT fire_flag)",
            len(df), len(filtered),
        )

        with tracker.step("SEM_FILTER(loss of vehicle control while driving)",
                          input_rows=len(filtered)) as st:
            hits = filtered.sem_filter(
                "The complaint {summary_text} describes a loss of vehicle "
                "control while driving"
            )
            st.set_output(hits)

        answer = int(len(hits))
        tracker.record("GROUP_BY([], count(*))", len(hits), 1)

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
