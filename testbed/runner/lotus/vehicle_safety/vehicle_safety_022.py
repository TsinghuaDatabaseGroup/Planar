#!/usr/bin/env python3
"""
vehicle_safety-022
Return the top three safety-event topics among complaints involving medical
attention and a concrete malfunction or crash sequence.
DAG: SCAN_TABLE(complaints) -> FILTER(medical attention) ->
     SEM_FILTER(concrete safety incident) -> SEM_CLASSIFY(topic) ->
     GROUP_BY(topic) -> ORDER_BY(count DESC, topic ASC) -> LIMIT(3)
Output: ordered list, metric: ordered_f1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, df_records, load_table, save_output, setup

TASK_ID = "vehicle_safety-022"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["medical_attention_flag", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        medical = complaints[complaints["medical_attention_flag"]]
        tracker.record(
            "FILTER(medical_attention_flag='true')",
            len(complaints),
            len(medical),
        )

        with tracker.step(
            "SEM_FILTER(concrete malfunction, crash sequence, or safety incident)",
            input_rows=len(medical),
        ) as step:
            concrete = medical.sem_filter(
                "The complaint {summary_text} describes a concrete vehicle "
                "malfunction, crash sequence, or safety incident rather than only "
                "a billing, paperwork, or service-scheduling issue."
            )
            step.set_output(concrete)

        with tracker.step(
            "SEM_CLASSIFY(primary_safety_event_topic)", input_rows=len(concrete)
        ) as step:
            classified = concrete.sem_map(
                "Classify the primary safety-event topic in complaint "
                "{summary_text}. Output exactly one label from: steering, braking, "
                "airbag, electrical, powertrain, other.",
                suffix="topic",
            )
            classified["topic"] = (
                classified["topic"]
                .astype(str)
                .str.strip()
                .str.strip(".")
                .str.lower()
            )
            step.set_output(classified)

        grouped = (
            classified.groupby("topic", as_index=False, dropna=False)
            .size()
            .rename(columns={"size": "complaint_count"})
        )
        tracker.record(
            "GROUP_BY([topic], complaint_count=count(*))",
            len(classified),
            len(grouped),
        )

        ordered = grouped.sort_values(
            ["complaint_count", "topic"], ascending=[False, True]
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([complaint_count DESC, topic ASC])",
            len(grouped),
            len(ordered),
        )

        top_three = ordered.head(3).reset_index(drop=True)
        tracker.record("LIMIT(3)", len(ordered), len(top_three))
        answer = df_records(top_three[["topic", "complaint_count"]])

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
