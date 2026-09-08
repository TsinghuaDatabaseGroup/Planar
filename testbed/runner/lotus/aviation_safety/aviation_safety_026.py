#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-026."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    normalize_free_label,
    parse_record_list,
    save_output,
    setup,
    stable_mode,
)

TASK_ID = "aviation_safety-026"
MEASUREMENT_TYPES = ("altitude", "speed", "heading", "runway")


def numeric_value(value) -> str | None:
    if isinstance(value, (dict, list, tuple, set)) or value is None:
        return None
    text = " ".join(str(value).strip().strip("`\"'").split())
    return text if text and any(character.isdigit() for character in text) else None


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "state_reference"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[incidents["state_reference"] == "US"].copy()
        tracker.record(
            "FILTER(state_reference='US')",
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
            "SEM_FILTER(matched numeric expected-versus-actual comparison)",
            input_rows=len(reports),
        ) as step:
            candidates = reports.sem_filter(
                "The report {text} contains at least one numeric "
                "cleared-versus-flown or expected-versus-actual comparison for "
                "altitude, speed, heading, or runway, with both values belonging "
                "to the same comparison"
            )
            step.set_output(candidates)

        with tracker.step(
            "SEM_EXTRACT(distinct numeric comparisons)",
            input_rows=len(candidates),
        ) as step:
            raw = candidates.sem_extract(
                input_cols=["text"],
                output_cols={
                    "comparisons": (
                        "a JSON array with one object per distinct comparison. "
                        "Each object must contain measurement_type as exactly one "
                        "of altitude, speed, heading, runway; expected_value and "
                        "actual_value as separate numeric strings with their stated "
                        "units or designators; and deviation as a concise phrase of "
                        "at most six words. Do not pair unrelated numeric mentions"
                    )
                },
            )
            comparison_rows = []
            for report in raw.to_dict(orient="records"):
                for comparison in parse_record_list(report.get("comparisons")):
                    measurement_type = normalize_enum(
                        comparison.get("measurement_type"),
                        MEASUREMENT_TYPES,
                    )
                    expected_value = numeric_value(
                        comparison.get("expected_value")
                    )
                    actual_value = numeric_value(comparison.get("actual_value"))
                    if (
                        measurement_type is None
                        or expected_value is None
                        or actual_value is None
                    ):
                        continue
                    comparison_rows.append(
                        {
                            "incident_id": report["incident_id"],
                            "measurement_type": measurement_type,
                            "expected_value": expected_value,
                            "actual_value": actual_value,
                            "deviation": normalize_free_label(
                                comparison.get("deviation")
                            ),
                        }
                    )
            extracted = pd.DataFrame.from_records(
                comparison_rows,
                columns=[
                    "incident_id",
                    "measurement_type",
                    "expected_value",
                    "actual_value",
                    "deviation",
                ],
            ).drop_duplicates(
                subset=[
                    "incident_id",
                    "measurement_type",
                    "expected_value",
                    "actual_value",
                ]
            )
            extracted = extracted.reset_index(drop=True)
            step.set_output(extracted)

        grouped = (
            extracted.groupby("measurement_type", sort=False)
            .agg(
                comparison_count=("incident_id", "size"),
                most_common_deviation=("deviation", stable_mode),
            )
            .reset_index()
        )
        measurement_order = {
            label: index for index, label in enumerate(MEASUREMENT_TYPES)
        }
        grouped = grouped.sort_values(
            "measurement_type",
            key=lambda values: values.map(measurement_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([measurement_type], COUNT_DISTINCT(comparison), MODE)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
