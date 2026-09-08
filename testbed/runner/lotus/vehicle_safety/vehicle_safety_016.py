#!/usr/bin/env python3
"""
vehicle_safety-016
Compute the fraction of engine stall or power-loss complaints that happened
while the vehicle was moving or in traffic, rounded to three decimal places.
DAG: SCAN_TABLE(complaints) -> FILTER(component_id='ENGINE') ->
     SEM_FILTER(engine stall or power loss) -> SEM_CLASSIFY(event context) ->
     GROUP_BY([], counts) -> PROJECT(ratio)
Output: scalar, metric: rae
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, load_table, save_output, setup

TASK_ID = "vehicle_safety-016"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        engine = complaints[complaints["component_id"] == "ENGINE"]
        tracker.record(
            "FILTER(component_id='ENGINE')", len(complaints), len(engine)
        )

        with tracker.step(
            "SEM_FILTER(engine stall, shutoff, power loss, or failure to accelerate)",
            input_rows=len(engine),
        ) as step:
            relevant = engine.sem_filter(
                "The complaint {summary_text} describes an engine stall, engine "
                "shutoff, loss of power, or failure to accelerate caused by engine "
                "operation."
            )
            step.set_output(relevant)

        with tracker.step(
            "SEM_CLASSIFY(event_context)", input_rows=len(relevant)
        ) as step:
            classified = relevant.sem_map(
                "Classify the event context in complaint {summary_text}. Output "
                "exactly one label: moving_or_in_traffic when the event happened "
                "while the vehicle was moving, driving, or in traffic; otherwise "
                "output parked_idling_or_unclear for parked, idling, starting, or "
                "unclear situations.",
                suffix="event_context",
            )
            classified["event_context"] = (
                classified["event_context"].astype(str).str.strip().str.strip(".").str.lower()
            )
            step.set_output(classified)

        total = len(classified)
        moving_count = int(
            (classified["event_context"] == "moving_or_in_traffic").sum()
        )
        tracker.record(
            "GROUP_BY([], total=count(*), moving=count_if(event_context='moving_or_in_traffic'))",
            len(classified),
            1,
        )

        answer = round(moving_count / total, 3)
        tracker.record("PROJECT(ratio=round(moving/total, 3))", 1, 1)

    print(f"Result: {answer:.3f}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
