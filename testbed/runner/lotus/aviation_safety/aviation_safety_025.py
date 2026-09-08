#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-025."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-025"
MECHANISMS = (
    "data_entry_or_programming",
    "alert_interpretation",
    "automation_disengagement",
    "unexpected_capture_or_leveloff",
    "mode_awareness",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "primary_problem"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[
            incidents["primary_problem"] == "Human Factors"
        ].copy()
        tracker.record(
            "FILTER(primary_problem='Human Factors')",
            len(incidents),
            len(filtered),
            output=filtered,
        )

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(automation misunderstanding materially contributed)",
            input_rows=len(reports),
        ) as step:
            automation_related = reports.sem_filter(
                "The report {text} describes misunderstanding or surprise about "
                "automation that materially contributed to the event"
            )
            step.set_output(automation_related)

        with tracker.step(
            "SEM_FILTER(not equipment failure or ordinary manual-flying error)",
            input_rows=len(automation_related),
        ) as step:
            eligible = automation_related.sem_filter(
                "The primary issue in the report {text} was not an aircraft "
                "equipment failure and was not an ordinary manual-flying error "
                "without automation confusion"
            )
            step.set_output(eligible)

        with tracker.step(
            "SEM_EXTRACT(assign dominant automation-confusion mechanism)",
            input_rows=len(eligible),
        ) as step:
            classified = eligible.sem_extract(
                input_cols=["text"],
                output_cols={
                    "confusion_mechanism": (
                        "assign exactly one dominant mechanism using this "
                        "precedence: data_entry_or_programming, "
                        "alert_interpretation, automation_disengagement, "
                        "unexpected_capture_or_leveloff, mode_awareness. Return "
                        "not_applicable when the report does not clearly fit one"
                    )
                },
            )
            classified["confusion_mechanism"] = classified[
                "confusion_mechanism"
            ].map(lambda value: normalize_enum(value, MECHANISMS))
            assigned = classified.loc[
                classified["confusion_mechanism"].notna(),
                ["incident_id", "confusion_mechanism"],
            ].reset_index(drop=True)
            step.set_output(assigned)

        grouped = (
            assigned.groupby("confusion_mechanism", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        mechanism_order = {
            label: index for index, label in enumerate(MECHANISMS)
        }
        grouped = grouped.sort_values(
            "confusion_mechanism",
            key=lambda values: values.map(mechanism_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([confusion_mechanism], COUNT_DISTINCT)",
            len(assigned),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
