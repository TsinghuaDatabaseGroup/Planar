#!/usr/bin/env python3
"""
vehicle_safety-006
Among Subaru Outback complaints from the 2024 model year onward where no crash
was reported, how many distinct model years are represented in records
describing an unintended brake activation or sudden brake hold event?
DAG: SCAN_TABLE(complaints) -> FILTER(make='SUBARU' AND model='OUTBACK' AND
     model_year>=2024 AND NOT crash_flag) -> SEM_FILTER(unintended brake
     activation or sudden brake hold) -> GROUP_BY([], count_distinct(model_year))
Output: scalar (integer), metric: exact_match
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_table, save_output, StepTracker, Timer

TASK_ID = "vehicle_safety-006"

def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as t:
        df = load_table("nhtsa_vehicle_safety", "complaints.csv")
        tracker.record("SCAN_TABLE(complaints)", None, len(df))

        filtered = df[
            (df["make"] == "SUBARU")
            & (df["model"] == "OUTBACK")
            & (df["model_year"] >= 2024)
            & ~df["crash_flag"]
        ]
        tracker.record(
            "FILTER(make='SUBARU' AND model='OUTBACK' AND model_year>=2024 AND NOT crash_flag)",
            len(df), len(filtered),
        )

        with tracker.step("SEM_FILTER(unintended brake activation or sudden brake hold)",
                          input_rows=len(filtered)) as st:
            hits = filtered.sem_filter(
                "The complaint {summary_text} describes an unintended brake "
                "activation or sudden brake hold event"
            )
            st.set_output(hits)

        answer = int(hits["model_year"].nunique())
        tracker.record("GROUP_BY([], count_distinct(model_year))", len(hits), 1)

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
