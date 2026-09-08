#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-003."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    parse_bool,
    parse_label_list,
    result_frame,
    save_output,
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
SOURCE_CATEGORIES = (
    "electrical",
    "oil_or_hydraulic",
    "air_conditioning_or_bleed",
    "engine_or_apu",
    "unknown_or_other",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem", "result_summary"],
        )
        tracker.record("scan", None, incidents)

        filtered = incidents.loc[
            incidents["primary_problem"] == "Aircraft"
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(incidents),
            filtered,
        )

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "scan",
            len(filtered),
            reports,
        )

        semantic = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "qualifies",
                    "type": bool,
                    "desc": (
                        "True only when smoke, fumes, or odor prompted at least one "
                        "permitted crew response and no visible fire was reported."
                    ),
                },
                {
                    "name": "operational_responses",
                    "type": list[str],
                    "desc": (
                        "Every supported response directly prompted by the smoke, "
                        "fumes, or odor, selected only from oxygen_masks_used, "
                        "emergency_declared, diverted, returned, precautionary_landing; "
                        "return an empty list when none qualifies."
                    ),
                },
                {
                    "name": "suspected_source_category",
                    "type": str,
                    "desc": (
                        "The suspected source normalized to exactly one of electrical, "
                        "oil_or_hydraulic, air_conditioning_or_bleed, engine_or_apu, "
                        "unknown_or_other."
                    ),
                },
            ],
            desc=(
                "Identify crew responses caused by smoke, fumes, or odor when no "
                "visible fire was reported, and classify the suspected source. The "
                "structured result summary may establish emergency, diversion, or "
                "return only when the narrative confirms the smoke/fumes/odor link."
            ),
            depends_on=["result_summary", "text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        raw = result_frame(semantic_result)
        raw["qualifies"] = raw["qualifies"].map(parse_bool)
        raw["operational_responses"] = raw["operational_responses"].map(
            lambda value: parse_label_list(value, RESPONSES)
        )
        raw["suspected_source_category"] = raw[
            "suspected_source_category"
        ].map(lambda value: normalize_enum(value, SOURCE_CATEGORIES))

        rows = []
        for record in raw.itertuples(index=False):
            if not record.qualifies or record.suspected_source_category is None:
                continue
            for response in record.operational_responses:
                rows.append(
                    {
                        "incident_id": record.incident_id,
                        "operational_response": response,
                        "suspected_source_category": (
                            record.suspected_source_category
                        ),
                    }
                )
        extracted = pd.DataFrame.from_records(
            rows,
            columns=[
                "incident_id",
                "operational_response",
                "suspected_source_category",
            ],
        )
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            semantic_result,
            time.time() - started,
        )

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
        response_order = {
            label: index for index, label in enumerate(RESPONSES)
        }
        grouped = grouped.sort_values(
            "operational_response",
            key=lambda values: values.map(response_order),
        ).reset_index(drop=True)
        tracker.record(
            "groupby",
            len(extracted),
            grouped,
        )
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
