#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-002."""

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
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-002"
REASONS = (
    "late_clearance",
    "confusing_or_ambiguous",
    "workload_or_timing_pressure",
    "expectation_or_readback_error",
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

        reports = load_selected_texts("asrs", filtered)[["incident_id", "text"]]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(non-equipment clearance noncompliance reason)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "qualifies": (
                        "true only when a specific ATC clearance or restriction "
                        "could not be followed for a non-mechanical reason; false "
                        "for mechanical limitations and all other reports"
                    ),
                    "primary_reason": (
                        "when qualifies is true, the single primary reason, exactly "
                        "one of late_clearance, confusing_or_ambiguous, "
                        "workload_or_timing_pressure, "
                        "expectation_or_readback_error; otherwise not_applicable"
                    ),
                    "has_numeric_time": (
                        "true only if the report states a numeric amount of time "
                        "available to comply with that clearance or restriction"
                    ),
                    "has_numeric_distance": (
                        "true only if the report states a numeric amount of distance "
                        "available to comply with that clearance or restriction"
                    ),
                },
            )
            raw["qualifies"] = raw["qualifies"].map(parse_bool)
            raw["primary_reason"] = raw["primary_reason"].map(
                lambda value: normalize_enum(value, REASONS)
            )
            raw["has_numeric_time"] = raw["has_numeric_time"].map(parse_bool)
            raw["has_numeric_distance"] = raw["has_numeric_distance"].map(
                parse_bool
            )
            extracted = raw.loc[
                raw["qualifies"] & raw["primary_reason"].notna(),
                [
                    "incident_id",
                    "primary_reason",
                    "has_numeric_time",
                    "has_numeric_distance",
                ],
            ].reset_index(drop=True)
            step.set_output(extracted)

        grouped = (
            extracted.groupby("primary_reason", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        time_counts = (
            extracted[extracted["has_numeric_time"]]
            .groupby("primary_reason")["incident_id"]
            .nunique()
        )
        distance_counts = (
            extracted[extracted["has_numeric_distance"]]
            .groupby("primary_reason")["incident_id"]
            .nunique()
        )
        grouped["with_numeric_time_count"] = (
            grouped["primary_reason"].map(time_counts).fillna(0).astype(int)
        )
        grouped["with_numeric_distance_count"] = (
            grouped["primary_reason"].map(distance_counts).fillna(0).astype(int)
        )
        reason_order = {label: index for index, label in enumerate(REASONS)}
        grouped = grouped.sort_values(
            "primary_reason",
            key=lambda values: values.map(reason_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([primary_reason], COUNT_DISTINCT, COUNT_DISTINCT_IF)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
