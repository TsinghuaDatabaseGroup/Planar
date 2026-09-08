#!/usr/bin/env python3
"""
vehicle_safety-029
Compare average injury counts for Honda ADAS and BRAKES complaints describing
tangible safety events that occurred while the vehicle was being driven.
DAG: SCAN_TABLE -> FILTER -> SEM_FILTER(tangible) -> SEM_FILTER(in motion) ->
     GROUP_BY(component_id) -> PROJECT(label with 0.01 tolerance)
Output: label, metric: label_accuracy
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, load_table, save_output, setup

TASK_ID = "vehicle_safety-029"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["make", "component_id", "injury_count", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        selected = complaints[
            (complaints["make"] == "HONDA")
            & complaints["component_id"].isin(["ADAS", "BRAKES"])
        ]
        tracker.record(
            "FILTER(make='HONDA' AND component_id IN ['ADAS','BRAKES'])",
            len(complaints),
            len(selected),
        )

        with tracker.step(
            "SEM_FILTER(tangible safety event, not minor inconvenience)",
            input_rows=len(selected),
        ) as step:
            tangible = selected.sem_filter(
                "The complaint {summary_text} describes a tangible safety event, "
                "not a minor inconvenience such as screen flicker, infotainment "
                "lag, or a cosmetic blemish."
            )
            step.set_output(tangible)

        with tracker.step(
            "SEM_FILTER(problem manifested while vehicle was being driven)",
            input_rows=len(tangible),
        ) as step:
            in_motion = tangible.sem_filter(
                "The complaint {summary_text} indicates that the problem "
                "manifested while the vehicle was being driven, not while it was "
                "parked or stationary."
            )
            step.set_output(in_motion)

        grouped = in_motion.groupby("component_id", as_index=False).agg(
            avg_injury=("injury_count", "mean")
        )
        tracker.record(
            "GROUP_BY([component_id], avg_injury=avg(injury_count))",
            len(in_motion),
            len(grouped),
        )

        averages = dict(
            zip(grouped["component_id"], grouped["avg_injury"], strict=True)
        )
        adas_avg = float(averages.get("ADAS", float("nan")))
        brakes_avg = float(averages.get("BRAKES", float("nan")))
        if adas_avg - brakes_avg > 0.01:
            answer = "ADAS_higher"
        elif brakes_avg - adas_avg > 0.01:
            answer = "BRAKES_higher"
        else:
            answer = "tie"

        tracker.record(
            "PROJECT(label=compare(avg_injury_ADAS, avg_injury_BRAKES, tolerance=0.01))",
            len(grouped),
            1,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
