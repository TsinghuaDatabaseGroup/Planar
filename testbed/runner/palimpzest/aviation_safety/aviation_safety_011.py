#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-011."""

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
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-011"
SYSTEM_CATEGORIES = (
    "engine",
    "pressurization_or_bleed_or_pack",
    "hydraulic",
    "flight_control",
    "not_target",
)
RESPONSE_CATEGORIES = (
    "return_to_departure",
    "emergency_or_divert",
    "continue_after_troubleshooting",
    "maintenance_action_only",
)


def response_category(value) -> str | None:
    summary = "" if pd.isna(value) else str(value)
    if "Flight Crew Returned To Departure Airport" in summary:
        return "return_to_departure"
    if any(
        label in summary
        for label in (
            "General Declared Emergency",
            "Flight Crew Diverted",
            "Flight Crew Landed in Emergency Condition",
        )
    ):
        return "emergency_or_divert"
    if "Flight Crew Overcame Equipment Problem" in summary:
        return "continue_after_troubleshooting"
    if "General Maintenance Action" in summary:
        return "maintenance_action_only"
    return None


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
                "far_part",
                "anomaly_summary",
                "result_summary",
                "flight_phase",
            ],
        )
        tracker.record("scan", None, incidents)

        equipment = incidents.loc[
            incidents["anomaly_summary"].str.contains(
                "Aircraft Equipment Problem", na=False
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), equipment)

        reports = load_selected_texts("asrs", equipment)[
            [
                "incident_id",
                "far_part",
                "anomaly_summary",
                "result_summary",
                "flight_phase",
                "text",
            ]
        ]
        tracker.record("scan", len(equipment), reports)

        semantic = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "system_category",
                    "type": str,
                    "desc": (
                        "Exactly one of engine, "
                        "pressurization_or_bleed_or_pack, hydraulic, "
                        "flight_control, or not_target. Use engine for an "
                        "airborne engine or powerplant problem; "
                        "pressurization_or_bleed_or_pack for cabin pressure, "
                        "pneumatic bleed-air, or air-conditioning-pack problems; "
                        "hydraulic for hydraulic-system problems; flight_control "
                        "for flight-control-system problems; and not_target for "
                        "ground-only problems or any other system."
                    ),
                }
            ],
            desc=(
                "Classify the aircraft-equipment problem by system using the "
                "reported flight phase, anomaly summary, and narrative."
            ),
            depends_on=["flight_phase", "anomaly_summary", "text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        classified = result_frame(semantic_result)
        classified["system_category"] = classified[
            "system_category"
        ].map(lambda value: normalize_enum(value, SYSTEM_CATEGORIES))
        tracker.record_semantic(
            "sem_map",
            len(reports),
            classified,
            semantic_result,
            time.time() - started,
        )

        targeted = classified.loc[
            (classified["far_part"] == "Part 121")
            & classified["anomaly_summary"].str.contains(
                "Aircraft Equipment Problem Critical", na=False
            )
            & classified["system_category"].notna()
            & (classified["system_category"] != "not_target")
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), targeted)

        projected = targeted[
            ["incident_id", "system_category", "result_summary"]
        ].copy()
        projected["response_category"] = projected["result_summary"].map(
            response_category
        )
        projected = projected[
            ["incident_id", "system_category", "response_category"]
        ]
        tracker.record("project", len(targeted), projected)

        responded = projected.loc[
            projected["response_category"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(projected), responded)

        grouped = (
            responded.groupby(
                ["system_category", "response_category"], sort=False
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        system_order = {
            label: index for index, label in enumerate(SYSTEM_CATEGORIES)
        }
        response_order = {
            label: index for index, label in enumerate(RESPONSE_CATEGORIES)
        }
        grouped = grouped.sort_values(
            ["system_category", "response_category"],
            key=lambda values: (
                values.map(system_order)
                if values.name == "system_category"
                else values.map(response_order)
            ),
        ).reset_index(drop=True)
        tracker.record("groupby", len(responded), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
