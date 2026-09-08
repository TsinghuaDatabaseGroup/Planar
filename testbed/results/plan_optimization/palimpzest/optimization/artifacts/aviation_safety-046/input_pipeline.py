#!/usr/bin/env python3
"""Plan-optimization pipeline for aviation_safety-046."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    run_plan_optimization,
    save_output,
)

TASK_ID = "aviation_safety-046"
EVENT_TYPES = (
    "equipment_failure",
    "airborne_conflict",
    "ground_conflict",
    "clearance_deviation",
    "weather_event",
    "maintenance_event",
    "cabin_event",
    "other",
)
BLOCK_KEYS = [
    "incident_date",
    "state_reference",
    "locale_reference_type",
    "primary_problem",
]
STRUCTURED_FIELDS = ("flight_phase", "anomaly_summary", "result_summary")


def normalize_left_signature(record: dict) -> dict:
    return {
        "normalized_left_event_type": normalize_enum(
            record.get("left_primary_event_type"), EVENT_TYPES
        )
    }


def normalize_right_signature(record: dict) -> dict:
    return {
        "normalized_right_event_type": normalize_enum(
            record.get("right_primary_event_type"), EVENT_TYPES
        )
    }


def valid_left_signature(record: dict) -> bool:
    return bool(record.get("normalized_left_event_type"))


def valid_right_signature(record: dict) -> bool:
    return bool(record.get("normalized_right_event_type"))


def same_blocked_pair(record: dict) -> bool:
    return (
        record.get("left_incident_id") == record.get("left_incident_id_right")
        and record.get("right_incident_id")
        == record.get("right_incident_id_right")
    )


def values_equal(left, right) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if left_missing and right_missing:
        return True
    if left_missing or right_missing:
        return False
    return left == right


def add_output_fields(record: dict) -> dict:
    differing = [
        field
        for field in STRUCTURED_FIELDS
        if not values_equal(
            record.get(f"left_{field}"),
            record.get(f"right_{field}"),
        )
    ]
    return {
        "shared_primary_event_type": record.get("normalized_left_event_type"),
        "differing_structured_fields": differing,
    }


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)
    incident_columns = [
        "incident_id",
        "incident_date",
        "state_reference",
        "locale_reference_type",
        "primary_problem",
        "flight_phase",
        "anomaly_summary",
        "result_summary",
        "text_file",
    ]

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv", incident_columns)
        left = incidents.rename(
            columns={
                "incident_id": "left_incident_id",
                "flight_phase": "left_flight_phase",
                "anomaly_summary": "left_anomaly_summary",
                "result_summary": "left_result_summary",
                "text_file": "left_text_file",
            }
        ).dropna(subset=BLOCK_KEYS)
        right = incidents.rename(
            columns={
                "incident_id": "right_incident_id",
                "flight_phase": "right_flight_phase",
                "anomaly_summary": "right_anomaly_summary",
                "result_summary": "right_result_summary",
                "text_file": "right_text_file",
            }
        ).dropna(subset=BLOCK_KEYS)
        blocked_pairs = left.merge(
            right,
            on=BLOCK_KEYS,
            how="inner",
            validate="many_to_many",
        )
        blocked_pairs = blocked_pairs.loc[
            blocked_pairs["left_incident_id"]
            < blocked_pairs["right_incident_id"]
        ].reset_index(drop=True)

        left_documents = load_selected_texts(
            "asrs",
            blocked_pairs,
            path_column="left_text_file",
            output_column="left_text",
        )[[
            "left_incident_id",
            "right_incident_id",
            "incident_date",
            "left_flight_phase",
            "right_flight_phase",
            "left_anomaly_summary",
            "right_anomaly_summary",
            "left_result_summary",
            "right_result_summary",
            "left_text",
        ]]
        right_documents = load_selected_texts(
            "asrs",
            blocked_pairs,
            path_column="right_text_file",
            output_column="right_text",
        )[["left_incident_id", "right_incident_id", "right_text"]]

        left_signatures = memory_dataset(
            f"{TASK_ID}-left", left_documents
        ).sem_map(
            cols=[
                {
                    "name": "left_primary_event_type",
                    "type": str,
                    "desc": (
                        "Exactly one of equipment_failure, airborne_conflict, "
                        "ground_conflict, clearance_deviation, weather_event, "
                        "maintenance_event, cabin_event, or other."
                    ),
                },
                {
                    "name": "left_measurements",
                    "type": list[str],
                    "desc": "Distinctive measurements in the occurrence.",
                },
                {
                    "name": "left_action_sequence",
                    "type": list[str],
                    "desc": "The distinctive ordered action sequence.",
                },
                {
                    "name": "left_outcome",
                    "type": str,
                    "desc": "The reported outcome as a concise phrase.",
                },
            ],
            desc="Extract the normalized event signature from the left report.",
            depends_on=["left_incident_id", "right_incident_id", "left_text"],
        )
        left_signatures = left_signatures.map(
            normalize_left_signature,
            cols=[
                {
                    "name": "normalized_left_event_type",
                    "type": str | None,
                    "desc": "Validated left event type.",
                }
            ],
            depends_on=["left_primary_event_type"],
        ).filter(
            valid_left_signature,
            depends_on=["normalized_left_event_type"],
        )

        right_signatures = memory_dataset(
            f"{TASK_ID}-right", right_documents
        ).sem_map(
            cols=[
                {
                    "name": "right_primary_event_type",
                    "type": str,
                    "desc": (
                        "Exactly one of equipment_failure, airborne_conflict, "
                        "ground_conflict, clearance_deviation, weather_event, "
                        "maintenance_event, cabin_event, or other."
                    ),
                },
                {
                    "name": "right_measurements",
                    "type": list[str],
                    "desc": "Distinctive measurements in the occurrence.",
                },
                {
                    "name": "right_action_sequence",
                    "type": list[str],
                    "desc": "The distinctive ordered action sequence.",
                },
                {
                    "name": "right_outcome",
                    "type": str,
                    "desc": "The reported outcome as a concise phrase.",
                },
            ],
            desc="Extract the normalized event signature from the right report.",
            depends_on=["left_incident_id", "right_incident_id", "right_text"],
        )
        right_signatures = right_signatures.map(
            normalize_right_signature,
            cols=[
                {
                    "name": "normalized_right_event_type",
                    "type": str | None,
                    "desc": "Validated right event type.",
                }
            ],
            depends_on=["right_primary_event_type"],
        ).filter(
            valid_right_signature,
            depends_on=["normalized_right_event_type"],
        )

        plan = left_signatures.sem_join(
            right_signatures,
            condition=(
                "Compare only rows carrying the same left and right blocked-pair "
                "incident IDs. Match only when both narratives describe the "
                "same underlying occurrence, with a distinctive compatible "
                "event progression, measurements, actions, and outcome. Generic "
                "topical similarity is insufficient."
            ),
            depends_on=[
                "left_incident_id",
                "right_incident_id",
                "normalized_left_event_type",
                "left_measurements",
                "left_action_sequence",
                "left_outcome",
                "left_incident_id_right",
                "right_incident_id_right",
                "normalized_right_event_type",
                "right_measurements",
                "right_action_sequence",
                "right_outcome",
            ],
        )
        plan = plan.filter(
            same_blocked_pair,
            depends_on=[
                "left_incident_id",
                "right_incident_id",
                "left_incident_id_right",
                "right_incident_id_right",
            ],
        )
        plan = plan.map(
            add_output_fields,
            cols=[
                {
                    "name": "shared_primary_event_type",
                    "type": str,
                    "desc": "Shared normalized primary event type.",
                },
                {
                    "name": "differing_structured_fields",
                    "type": list[str],
                    "desc": "Names of differing structured fields.",
                },
            ],
            depends_on=[
                "normalized_left_event_type",
                "left_flight_phase",
                "right_flight_phase",
                "left_anomaly_summary",
                "right_anomaly_summary",
                "left_result_summary",
                "right_result_summary",
            ],
        )
        plan = plan.project(
            [
                "left_incident_id",
                "right_incident_id",
                "incident_date",
                "shared_primary_event_type",
                "differing_structured_fields",
            ]
        )

        started = time.time()
        optimized = run_plan_optimization(
            plan,
            config,
            task_id=TASK_ID,
            embedding_join_block_on=[
                "left_incident_id",
                "right_incident_id",
            ],
        )
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True)
        tracker.record_semantic(
            "optimized_plan",
            {"left": len(left_documents), "right": len(right_documents)},
            output,
            optimized.result,
            time.time() - started,
        )
        ordered = output.sort_values(
            ["incident_date", "left_incident_id", "right_incident_id"],
            ascending=[False, True, True],
            kind="stable",
        ).head(20).reset_index(drop=True)
        ordered.insert(0, "rank", range(1, len(ordered) + 1))
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
