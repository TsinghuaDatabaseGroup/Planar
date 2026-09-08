#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-035."""

from __future__ import annotations

import os
import sys
import time

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
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-035"
VISUAL_FACTORS = (
    "reduced_visibility",
    "inadequate_or_confusing_lighting",
    "glare",
    "missed_visual_scan",
    "obscured_signage_or_markings",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            [
                "incident_id",
                "text_file",
                "aircraft_operator",
                "locale_reference_type",
            ],
        )
        tracker.record("scan", None, incidents)

        carrier_airport = incidents.loc[
            (incidents["aircraft_operator"] == "Air Carrier")
            & (incidents["locale_reference_type"] == "Airport")
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), carrier_airport)

        reports = load_selected_texts("asrs", carrier_airport)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(carrier_airport), reports)

        factor_plan = memory_dataset(TASK_ID, reports).sem_flat_map(
            cols=[
                {
                    "name": "visual_factor",
                    "type": str,
                    "desc": (
                        "Exactly one of reduced_visibility, "
                        "inadequate_or_confusing_lighting, glare, "
                        "missed_visual_scan, or obscured_signage_or_markings."
                    ),
                }
            ],
            desc=(
                "Emit one row for every listed visual factor that materially "
                "contributed to the ground conflict. An incident may emit several "
                "different factors. Do not emit a factor that is merely mentioned "
                "or did not contribute to the event, and emit each factor at most "
                "once per incident."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        factor_result = factor_plan.run(config)
        extracted = result_frame(factor_result)
        extracted["visual_factor"] = extracted["visual_factor"].map(
            lambda value: normalize_enum(value, VISUAL_FACTORS)
        )
        tracker.record_semantic(
            "sem_flat_map",
            len(reports),
            extracted,
            factor_result,
            time.time() - started,
        )

        semantic_factors = extracted.loc[
            extracted["visual_factor"].notna(),
            ["incident_id", "visual_factor"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), semantic_factors)

        environment_attributes = load_table(
            "asrs",
            "environment_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, environment_attributes)

        light_rows = environment_attributes.loc[
            (environment_attributes["attribute"] == "light")
            & environment_attributes["value"].isin(
                ["Night", "Dusk", "Dawn", "Daylight"]
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(environment_attributes), light_rows)

        light_records = light_rows[["incident_id", "value"]].copy()
        light_records["light_group"] = light_records["value"].map(
            lambda value: (
                "low_light"
                if value in {"Night", "Dusk", "Dawn"}
                else "daylight"
            )
        )
        light_records = light_records[["incident_id", "light_group"]]
        tracker.record("project", len(light_rows), light_records)

        light_records = light_records.drop_duplicates().reset_index(drop=True)
        tracker.record("distinct", len(light_rows), light_records)

        factors_with_light = semantic_factors.merge(
            light_records,
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "join",
            {
                "left": len(semantic_factors),
                "right": len(light_records),
            },
            factors_with_light,
        )

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "label"],
        )
        tracker.record("scan", None, events)

        ground_rows = events.loc[
            (events["label"] == "Conflict Ground Conflict")
            | events["label"].str.startswith("Ground Incursion ", na=False)
            | events["label"].str.startswith("Ground Excursion ", na=False)
        ].reset_index(drop=True)
        tracker.record("filter", len(events), ground_rows)

        ground_events = ground_rows[["incident_id"]].drop_duplicates().reset_index(
            drop=True
        )
        tracker.record("distinct", len(ground_rows), ground_events)

        joined = factors_with_light.merge(
            ground_events,
            on="incident_id",
            how="inner",
            validate="many_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(factors_with_light),
                "right": len(ground_events),
            },
            joined,
        )

        grouped = (
            joined.groupby(["light_group", "visual_factor"], sort=True)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record("groupby", len(joined), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
