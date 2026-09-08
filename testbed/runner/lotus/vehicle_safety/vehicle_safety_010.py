#!/usr/bin/env python3
"""
vehicle_safety-010
Do at least ten Toyota RAV4 complaints from the 2021 through 2024 model years,
with at least one occupant injured and no fire reported, describe a loss of
braking ability or unexpected brake failure?
DAG: SCAN_TABLE(complaints) -> FILTER(make='TOYOTA' AND model='RAV4' AND
     model_year IN [2021,2022,2023,2024] AND injury_count>=1 AND NOT fire_flag)
     -> SEM_AGGREGATE(at least ten describe loss of braking -> Yes/No)
Output: label (Yes/No), metric: label_accuracy
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_table, save_output, clean_label, StepTracker, Timer

TASK_ID = "vehicle_safety-010"

def main():
    setup(max_tokens=512)
    tracker = StepTracker()

    with Timer() as t:
        df = load_table("nhtsa_vehicle_safety", "complaints.csv")
        tracker.record("SCAN_TABLE(complaints)", None, len(df))

        filtered = df[
            (df["make"] == "TOYOTA")
            & (df["model"] == "RAV4")
            & df["model_year"].isin([2021, 2022, 2023, 2024])
            & (df["injury_count"] >= 1)
            & ~df["fire_flag"]
        ]
        tracker.record(
            "FILTER(make='TOYOTA' AND model='RAV4' AND model_year IN [2021..2024] "
            "AND injury_count>=1 AND NOT fire_flag)",
            len(df), len(filtered),
        )

        with tracker.step("SEM_AGGREGATE(at least ten describe loss of braking -> Yes/No)",
                          input_rows=len(filtered)) as st:
            agg = filtered.sem_agg(
                "Among the input complaint {summary_text} summaries, determine "
                "whether at least ten describe a loss of braking ability or "
                "unexpected brake failure. Output exactly one word: Yes or No."
            )
            st.set_output(agg)

        answer = clean_label(agg["_output"].iloc[0])

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
