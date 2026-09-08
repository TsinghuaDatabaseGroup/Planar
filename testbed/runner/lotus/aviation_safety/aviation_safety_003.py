#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-003."""

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
    parse_label_list,
    save_output,
    setup,
    stable_mode,
)

TASK_ID = "aviation_safety-003"
RESPONSES = (
    "oxygen_masks_used",
    "emergency_declared",
    "diverted",
    "returned",
    "precautionary_landing",
)
SOURCES = (
    "electrical",
    "oil_or_hydraulic",
    "air_conditioning_or_bleed",
    "engine_or_apu",
    "unknown_or_other",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "text_file",
                "primary_problem",
                "result_summary",
            ]
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
            "SEM_EXTRACT(smoke response and suspected source)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["result_summary", "text"],
                output_cols={
                    "qualifies": (
                        "true only when smoke, fumes, or odor caused at least one "
                        "listed operational response and no visible fire was "
                        "reported; false otherwise"
                    ),
                    "operational_responses": (
                        "a JSON array containing every supported response caused by "
                        "the smoke, fumes, or odor, chosen only from "
                        "oxygen_masks_used, emergency_declared, diverted, returned, "
                        "precautionary_landing; use an empty array if none. The "
                        "structured result summary may support emergency, diversion, "
                        "or return only when consistent with the narrative"
                    ),
                    "suspected_source_category": (
                        "the single suspected source category, exactly one of "
                        "electrical, oil_or_hydraulic, air_conditioning_or_bleed, "
                        "engine_or_apu, unknown_or_other"
                    ),
                },
            )
            raw["qualifies"] = raw["qualifies"].map(parse_bool)
            raw["operational_responses"] = raw["operational_responses"].map(
                lambda value: parse_label_list(value, RESPONSES)
            )
            raw["suspected_source_category"] = raw[
                "suspected_source_category"
            ].map(lambda value: normalize_enum(value, SOURCES))
            qualified = raw.loc[
                raw["qualifies"]
                & raw["suspected_source_category"].notna()
                & raw["operational_responses"].map(bool),
                [
                    "incident_id",
                    "operational_responses",
                    "suspected_source_category",
                ],
            ]
            extracted = (
                qualified.explode("operational_responses")
                .rename(columns={"operational_responses": "operational_response"})
                .reset_index(drop=True)
            )
            step.set_output(extracted)

        grouped = (
            extracted.groupby("operational_response", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_suspected_source_category=(
                    "suspected_source_category",
                    stable_mode,
                ),
            )
            .reset_index()
        )
        response_order = {label: index for index, label in enumerate(RESPONSES)}
        grouped = grouped.sort_values(
            "operational_response",
            key=lambda values: values.map(response_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([operational_response], COUNT_DISTINCT, MODE)",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
