#!/usr/bin/env python3
"""
vehicle_safety-027
Classify injury-causing unintended Toyota airbag deployments by likely root
cause and return counts, percentages, and the dominant label.
DAG: SCAN_TABLE(complaints) -> FILTER(TOYOTA AIRBAG) ->
     SEM_FILTER(unintended deployment) -> SEM_FILTER(injury or burn) ->
     SEM_CLASSIFY(root cause) -> GROUP_BY(label) -> PROJECT(dictionary)
Output: dictionary, metric: f1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, load_table, save_output, setup

TASK_ID = "vehicle_safety-027"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        toyota_airbag = complaints[
            (complaints["make"] == "TOYOTA")
            & (complaints["component_id"] == "AIRBAG")
        ]
        tracker.record(
            "FILTER(make='TOYOTA' AND component_id='AIRBAG')",
            len(complaints),
            len(toyota_airbag),
        )

        with tracker.step(
            "SEM_FILTER(unintended airbag deployment without collision)",
            input_rows=len(toyota_airbag),
        ) as step:
            unintended = toyota_airbag.sem_filter(
                "The complaint {summary_text} describes an unintended airbag "
                "deployment in which the airbag deployed without a corresponding "
                "collision or impact."
            )
            step.set_output(unintended)

        with tracker.step(
            "SEM_FILTER(injury or burn caused by deployment)",
            input_rows=len(unintended),
        ) as step:
            injury_cases = unintended.sem_filter(
                "The complaint {summary_text} reports an occupant injury or burn "
                "caused by the airbag deployment."
            )
            step.set_output(injury_cases)

        with tracker.step(
            "SEM_CLASSIFY(root_cause_label)", input_rows=len(injury_cases)
        ) as step:
            classified = injury_cases.sem_map(
                "Classify the likely primary root cause described by complaint "
                "{summary_text}. Output exactly one label: sensor_misfire, "
                "software_logic_error, inflator_defect, or other.",
                suffix="root_cause_label",
            )
            classified["root_cause_label"] = (
                classified["root_cause_label"]
                .astype(str)
                .str.strip()
                .str.strip(".")
                .str.lower()
            )
            step.set_output(classified)

        grouped = (
            classified.groupby("root_cause_label", as_index=False, dropna=False)
            .size()
            .rename(columns={"size": "count"})
        )
        tracker.record(
            "GROUP_BY([root_cause_label], count=count(*))",
            len(classified),
            len(grouped),
        )

        total = int(grouped["count"].sum())
        if grouped.empty:
            dominant_label = None
        else:
            max_count = grouped["count"].max()
            dominant_label = sorted(
                grouped.loc[
                    grouped["count"] == max_count, "root_cause_label"
                ].astype(str)
            )[0]
        per_label_counts = {
            str(label): {"count": int(count)}
            for label, count in zip(grouped["root_cause_label"], grouped["count"])
        }
        per_label_percentages = {
            str(label): round(100.0 * int(count) / total, 1)
            for label, count in zip(grouped["root_cause_label"], grouped["count"])
        } if total else {}
        answer = {
            "dominant_label": dominant_label,
            "per_label_counts": per_label_counts,
            "per_label_percentages": per_label_percentages,
        }
        tracker.record(
            "PROJECT(format_as_dictionary, dominant_label, per_label_counts, per_label_percentages)",
            len(grouped),
            1,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
