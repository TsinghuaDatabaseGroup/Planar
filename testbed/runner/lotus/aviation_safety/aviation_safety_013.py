#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-013."""

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
    normalize_free_label,
    parse_bool,
    save_output,
    setup,
    stable_mode,
)

TASK_ID = "aviation_safety-013"
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
    setup(max_tokens=4096)
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

        filtered = incidents[incidents["primary_problem"] == "Aircraft"].copy()
        tracker.record(
            "FILTER(primary_problem='Aircraft')",
            len(incidents),
            len(filtered),
            output=filtered,
        )

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(engine symptom, immediate action, and outcome)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["result_summary", "text"],
                output_cols={
                    "qualifies": (
                        "true only for an engine thrust, power, oil, temperature, "
                        "vibration, flameout, or other engine-indication problem"
                    ),
                    "symptom_category": (
                        "when qualifies is true, exactly one of thrust_loss, "
                        "oil_indication, temperature, vibration, flameout, "
                        "other_engine_indication; otherwise not_applicable"
                    ),
                    "immediate_crew_action": (
                        "when qualifies is true, the crew's immediate response as a "
                        "concise lower_snake_case phrase of at most six words; "
                        "otherwise not_applicable"
                    ),
                    "primary_outcome": (
                        "when qualifies is true, the single primary outcome, exactly "
                        "one of emergency, diversion, return, continued_flight, "
                        "maintenance_only; otherwise not_applicable"
                    ),
                },
            )
            raw["qualifies"] = raw["qualifies"].map(parse_bool)
            raw["symptom_category"] = raw["symptom_category"].map(
                lambda value: normalize_enum(value, SYMPTOMS)
            )
            raw["immediate_crew_action"] = raw[
                "immediate_crew_action"
            ].map(normalize_free_label)
            raw["primary_outcome"] = raw["primary_outcome"].map(
                lambda value: normalize_enum(value, OUTCOMES)
            )
            extracted = raw.loc[
                raw["qualifies"]
                & raw["symptom_category"].notna()
                & raw["primary_outcome"].notna(),
                [
                    "incident_id",
                    "symptom_category",
                    "immediate_crew_action",
                    "primary_outcome",
                ],
            ].reset_index(drop=True)
            step.set_output(extracted)

        grouped = (
            extracted.groupby("symptom_category", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_immediate_action=(
                    "immediate_crew_action",
                    stable_mode,
                ),
                most_common_outcome=("primary_outcome", stable_mode),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([symptom_category], COUNT_DISTINCT, MODE)",
            len(extracted),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["incident_count", "symptom_category"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([incident_count DESC, symptom_category ASC])",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
