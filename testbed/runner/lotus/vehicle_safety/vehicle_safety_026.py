#!/usr/bin/env python3
"""
vehicle_safety-026
Classify relevant Toyota RAV4 body complaints into four glass, water-intrusion,
or visibility root-cause labels and return their distribution and top label.
DAG: SCAN_TABLE(complaints) -> FILTER(RAV4 BODY) -> SEM_FILTER(relevant defect)
     -> SEM_CLASSIFY(root cause) -> GROUP_BY(label) -> PROJECT(dictionary)
Output: dictionary, metric: f1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, load_table, save_output, setup

TASK_ID = "vehicle_safety-026"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["make", "model", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        rav4_body = complaints[
            (complaints["make"] == "TOYOTA")
            & (complaints["model"] == "RAV4")
            & (complaints["component_id"] == "BODY")
        ]
        tracker.record(
            "FILTER(make='TOYOTA' AND model='RAV4' AND component_id='BODY')",
            len(complaints),
            len(rav4_body),
        )

        with tracker.step(
            "SEM_FILTER(safety-relevant glass, window, or water-in-door defect)",
            input_rows=len(rav4_body),
        ) as step:
            relevant = rav4_body.sem_filter(
                "The complaint {summary_text} describes a safety-relevant glass, "
                "window, windshield, sunroof or moonroof, or water-in-door/window "
                "visibility defect."
            )
            step.set_output(relevant)

        with tracker.step(
            "SEM_CLASSIFY(root_cause_label)", input_rows=len(relevant)
        ) as step:
            classified = relevant.sem_map(
                "Classify the likely primary root cause described by complaint "
                "{summary_text}. Output exactly one label: "
                "spontaneous_glass_breakage, door_water_intrusion, "
                "visibility_distortion_or_defect, or other.",
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

        if grouped.empty:
            most_common_label = None
        else:
            max_count = grouped["count"].max()
            most_common_label = sorted(
                grouped.loc[
                    grouped["count"] == max_count, "root_cause_label"
                ].astype(str)
            )[0]
        per_label_counts = {
            str(label): {"count": int(count)}
            for label, count in zip(grouped["root_cause_label"], grouped["count"])
        }
        answer = {
            "most_common_label": most_common_label,
            "per_label_counts": per_label_counts,
        }
        tracker.record(
            "PROJECT(format_as_dictionary, most_common_label, per_label_counts)",
            len(grouped),
            1,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
