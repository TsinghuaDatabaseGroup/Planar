#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-011."""

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

TASK_ID = "aviation_safety-011"
SYSTEMS = (
    "engine",
    "pressurization_or_bleed_or_pack",
    "hydraulic",
    "flight_control",
)
SYSTEM_LABELS = (*SYSTEMS, "not_target")
RESPONSES = (
    "return_to_departure",
    "emergency_or_divert",
    "continue_after_troubleshooting",
    "maintenance_action_only",
)


def response_category(result_summary):
    summary = result_summary if isinstance(result_summary, str) else ""
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


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "text_file",
                "far_part",
                "anomaly_summary",
                "result_summary",
                "flight_phase",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        equipment = incidents[
            incidents["anomaly_summary"].str.contains(
                "Aircraft Equipment Problem",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(anomaly_summary CONTAINS 'Aircraft Equipment Problem')",
            len(incidents),
            len(equipment),
            output=equipment,
        )

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
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(equipment),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(assign airborne equipment system category)",
            input_rows=len(reports),
        ) as step:
            classified = reports.sem_extract(
                input_cols=["flight_phase", "anomaly_summary", "text"],
                output_cols={
                    "system_category": (
                        "assign exactly one of engine, "
                        "pressurization_or_bleed_or_pack, hydraulic, "
                        "flight_control, not_target. Use not_target for a "
                        "ground-only problem or a problem involving another system"
                    )
                },
            )
            classified["system_category"] = classified[
                "system_category"
            ].map(lambda value: normalize_enum(value, SYSTEM_LABELS))
            classified["system_category"] = classified[
                "system_category"
            ].fillna("not_target")
            classified = classified[
                [
                    "incident_id",
                    "far_part",
                    "anomaly_summary",
                    "result_summary",
                    "system_category",
                ]
            ].reset_index(drop=True)
            step.set_output(classified)

        targeted = classified[
            (classified["far_part"] == "Part 121")
            & classified["anomaly_summary"].str.contains(
                "Aircraft Equipment Problem Critical",
                regex=False,
                na=False,
            )
            & (classified["system_category"] != "not_target")
        ].copy()
        tracker.record(
            "FILTER(Part 121 AND critical equipment problem AND system target)",
            len(classified),
            len(targeted),
            output=targeted,
        )

        projected = targeted[["incident_id", "system_category"]].copy()
        projected["response_category"] = targeted["result_summary"].map(
            response_category
        )
        tracker.record(
            "PROJECT(system_category, response_category=CASE result_summary)",
            len(targeted),
            len(projected),
            output=projected,
        )

        with_response = projected[
            projected["response_category"].notna()
        ].copy()
        tracker.record(
            "FILTER(response_category IS NOT NULL)",
            len(projected),
            len(with_response),
            output=with_response,
        )

        grouped = (
            with_response.groupby(
                ["system_category", "response_category"],
                sort=False,
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        system_order = {label: index for index, label in enumerate(SYSTEMS)}
        response_order = {
            label: index for index, label in enumerate(RESPONSES)
        }
        grouped = (
            grouped.assign(
                _system_order=grouped["system_category"].map(system_order),
                _response_order=grouped["response_category"].map(
                    response_order
                ),
            )
            .sort_values(["_system_order", "_response_order"])
            .drop(columns=["_system_order", "_response_order"])
            .reset_index(drop=True)
        )
        tracker.record(
            "GROUP_BY([system_category, response_category], COUNT_DISTINCT)",
            len(with_response),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
