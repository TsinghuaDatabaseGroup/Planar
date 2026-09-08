#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-018."""

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
    stable_mode,
)


TASK_ID = "aviation_safety-018"
SYMPTOMS = (
    "thrust_loss",
    "oil_indication",
    "temperature",
    "vibration",
    "flameout",
    "other_engine_indication",
)
OUTCOMES = (
    "emergency",
    "diversion",
    "return",
    "continued_flight",
    "maintenance_only",
)


def main():
    setup(max_tokens=512)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "primary_problem", "result_summary"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        aircraft_primary = incidents[
            incidents["primary_problem"] == "Aircraft"
        ].copy()
        tracker.record(
            "FILTER(primary_problem='Aircraft')",
            len(incidents),
            len(aircraft_primary),
            output=aircraft_primary,
        )

        reports = load_selected_texts("asrs", aircraft_primary)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(aircraft_primary),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(engine symptom category and primary outcome)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["result_summary", "text"],
                output_cols={
                    "symptom_category": (
                        "For a report about an engine thrust, power, oil, "
                        "temperature, vibration, or flameout problem, output "
                        "exactly one primary category from thrust_loss, "
                        "oil_indication, temperature, vibration, flameout, "
                        "other_engine_indication. For a non-engine problem, "
                        "output not_applicable."
                    ),
                    "primary_outcome": (
                        "For a qualifying engine problem, output exactly one "
                        "primary outcome from emergency, diversion, return, "
                        "continued_flight, maintenance_only. Otherwise output "
                        "not_applicable."
                    ),
                },
            )
            raw["symptom_category"] = raw["symptom_category"].map(
                lambda value: normalize_enum(value, SYMPTOMS)
            )
            raw["primary_outcome"] = raw["primary_outcome"].map(
                lambda value: normalize_enum(value, OUTCOMES)
            )
            extracted = raw.loc[
                raw["symptom_category"].notna()
                & raw["primary_outcome"].notna(),
                ["incident_id", "symptom_category", "primary_outcome"],
            ].reset_index(drop=True)
            step.set_output(extracted)

        grouped = (
            extracted.groupby("symptom_category", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_outcome=("primary_outcome", stable_mode),
            )
            .reset_index()
        )
        grouped = grouped.sort_values(
            ["incident_count", "symptom_category"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY(symptom_category, COUNT_DISTINCT, MODE)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        tracker.record(
            "ORDER_BY(incident_count DESC, symptom_category ASC)",
            len(grouped),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
