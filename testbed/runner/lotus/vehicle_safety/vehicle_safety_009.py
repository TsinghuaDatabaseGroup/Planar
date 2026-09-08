#!/usr/bin/env python3
"""
vehicle_safety-009
Among Subaru Outback complaints from the 2024 model year onward where no crash
was reported, which failure mode is dominant: electrical, mechanical, or
software? Output 'mixed' when the dominant mode is unclear or tied.
DAG: SCAN_TABLE(complaints) -> FILTER(make='SUBARU' AND model='OUTBACK' AND
     model_year>=2024 AND NOT crash_flag) -> SEM_AGGREGATE(dominant failure
     mode: electrical/mechanical/software/mixed)
Output: label, metric: label_accuracy
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_table, save_output, clean_label, StepTracker, Timer

TASK_ID = "vehicle_safety-009"

def main():
    setup(max_tokens=512)
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

        with tracker.step("SEM_AGGREGATE(dominant failure mode)", input_rows=len(filtered)) as st:
            agg = filtered.sem_agg(
                "Determine the dominant failure mode across all input complaint "
                "{summary_text} summaries: electrical, mechanical, or software. "
                "If unclear or tied, output mixed. Output exactly one word: "
                "electrical, mechanical, software, or mixed."
            )
            st.set_output(agg)

        answer = clean_label(agg["_output"].iloc[0])

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
