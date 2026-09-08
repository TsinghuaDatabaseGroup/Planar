#!/usr/bin/env python3
"""
vehicle_safety-017
Compute the fraction of unintended airbag deployments that happened before
collision impact or without a collision, rounded to three decimal places.
DAG: SCAN_TABLE(complaints) -> FILTER(component_id='AIRBAG') ->
     SEM_FILTER(unintended deployment) -> SEM_CLASSIFY(deployment timing) ->
     GROUP_BY([], counts) -> PROJECT(ratio)
Output: scalar, metric: rae
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, load_table, save_output, setup

TASK_ID = "vehicle_safety-017"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        airbags = complaints[complaints["component_id"] == "AIRBAG"]
        tracker.record(
            "FILTER(component_id='AIRBAG')", len(complaints), len(airbags)
        )

        with tracker.step(
            "SEM_FILTER(unintended or unexpected airbag deployment)",
            input_rows=len(airbags),
        ) as step:
            unintended = airbags.sem_filter(
                "The complaint {summary_text} describes an unintended or "
                "unexpected airbag deployment."
            )
            step.set_output(unintended)

        with tracker.step(
            "SEM_CLASSIFY(deployment_timing)", input_rows=len(unintended)
        ) as step:
            classified = unintended.sem_map(
                "Classify the deployment timing in complaint {summary_text}. "
                "Output exactly one label: pre_impact_or_no_collision when the "
                "airbag deployed before any collision impact or when no collision "
                "occurred; output during_or_after_collision when it deployed during "
                "or after a crash impact.",
                suffix="deployment_timing",
            )
            classified["deployment_timing"] = (
                classified["deployment_timing"]
                .astype(str)
                .str.strip()
                .str.strip(".")
                .str.lower()
            )
            step.set_output(classified)

        total = len(classified)
        pre_impact_or_no_collision_count = int(
            (
                classified["deployment_timing"]
                == "pre_impact_or_no_collision"
            ).sum()
        )
        tracker.record(
            "GROUP_BY([], total=count(*), pre_impact_or_no_collision=count_if(deployment_timing='pre_impact_or_no_collision'))",
            len(classified),
            1,
        )

        answer = round(pre_impact_or_no_collision_count / total, 3)
        tracker.record("PROJECT(ratio=round(pre_impact_or_no_collision/total, 3))", 1, 1)

    print(f"Result: {answer:.3f}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
