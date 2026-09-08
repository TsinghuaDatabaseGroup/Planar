#!/usr/bin/env python3
"""
vehicle_safety-024
For tangible 2024-2025 safety complaints, aggregate count, towing rate,
average injuries, and dominant extracted failure mode by component.
DAG: SCAN_TABLE(complaints) -> FILTER(model year) ->
     SEM_FILTER(tangible safety event) -> SEM_EXTRACT(failure_mode) ->
     GROUP_BY(component_id) -> PROJECT
Output: table, metric: table_f1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, df_records, load_table, save_output, setup

TASK_ID = "vehicle_safety-024"


def alpha_ascending_mode(values):
    counts = values.value_counts()
    max_count = counts.max()
    return sorted(str(value) for value in counts[counts == max_count].index)[0]


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            [
                "model_year",
                "component_id",
                "vehicle_towed_flag",
                "injury_count",
                "summary_text",
            ]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        recent = complaints[complaints["model_year"].isin([2024, 2025])]
        tracker.record(
            "FILTER(model_year IN [2024, 2025])",
            len(complaints),
            len(recent),
        )

        with tracker.step(
            "SEM_FILTER(tangible safety event)", input_rows=len(recent)
        ) as step:
            tangible = recent.sem_filter(
                "The complaint {summary_text} describes a tangible safety event, "
                "not only a paperwork issue, recall notice, cosmetic concern, or "
                "routine service request."
            )
            step.set_output(tangible)

        with tracker.step(
            "SEM_EXTRACT(failure_mode)", input_rows=len(tangible)
        ) as step:
            extracted = tangible.sem_extract(
                input_cols=["summary_text"],
                output_cols={
                    "failure_mode": (
                        "A short canonical lower-case failure-mode label describing "
                        "the primary failure in the complaint summary."
                    )
                },
            )
            extracted["failure_mode"] = (
                extracted["failure_mode"]
                .astype(str)
                .str.strip()
                .str.strip(".")
                .str.lower()
            )
            step.set_output(extracted)

        grouped = (
            extracted.groupby("component_id", as_index=False, dropna=False)
            .agg(
                complaint_count=("component_id", "size"),
                towed_rate=("vehicle_towed_flag", "mean"),
                avg_injury=("injury_count", "mean"),
                dominant_failure_mode=("failure_mode", alpha_ascending_mode),
            )
        )
        tracker.record(
            "GROUP_BY([component_id], complaint_count, towed_rate, avg_injury, mode_with_tiebreak(failure_mode))",
            len(extracted),
            len(grouped),
        )

        result = grouped[
            [
                "component_id",
                "complaint_count",
                "towed_rate",
                "avg_injury",
                "dominant_failure_mode",
            ]
        ]
        tracker.record(
            "PROJECT([component_id, complaint_count, towed_rate, avg_injury, dominant_failure_mode])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
