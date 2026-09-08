#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-046."""

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
JOIN_INPUT_COLUMNS = [
    "left_incident_id",
    "right_incident_id",
    "left_primary_event_type",
    "left_measurements",
    "left_action_sequence",
    "left_outcome",
    "left_incident_id_right",
    "right_incident_id_right",
    "right_primary_event_type",
    "right_measurements",
    "right_action_sequence",
    "right_outcome",
]
MATCH_COLUMNS = [
    *JOIN_INPUT_COLUMNS,
    "incident_date",
    "left_flight_phase",
    "right_flight_phase",
    "left_anomaly_summary",
    "right_anomaly_summary",
    "left_result_summary",
    "right_result_summary",
]


def values_equal(left, right) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if left_missing and right_missing:
        return True
    if left_missing or right_missing:
        return False
    return left == right


def differing_fields(row) -> list[str]:
    return [
        field
        for field in STRUCTURED_FIELDS
        if not values_equal(row[f"left_{field}"], row[f"right_{field}"])
    ]


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)
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
        left_incidents = load_table("asrs", "incidents.csv", incident_columns)
        tracker.record("scan", None, left_incidents)

        right_incidents = load_table("asrs", "incidents.csv", incident_columns)
        tracker.record("scan", None, right_incidents)

        left_records = left_incidents.rename(
            columns={
                "incident_id": "left_incident_id",
                "flight_phase": "left_flight_phase",
                "anomaly_summary": "left_anomaly_summary",
                "result_summary": "left_result_summary",
                "text_file": "left_text_file",
            }
        )
        right_records = right_incidents.rename(
            columns={
                "incident_id": "right_incident_id",
                "flight_phase": "right_flight_phase",
                "anomaly_summary": "right_anomaly_summary",
                "result_summary": "right_result_summary",
                "text_file": "right_text_file",
            }
        )
        left_records = left_records.dropna(subset=BLOCK_KEYS)
        right_records = right_records.dropna(subset=BLOCK_KEYS)
        join_candidates = left_records.merge(
            right_records,
            on=BLOCK_KEYS,
            how="inner",
            validate="many_to_many",
        )
        blocked_pairs = join_candidates.loc[
            join_candidates["left_incident_id"]
            < join_candidates["right_incident_id"]
        ].reset_index(drop=True)
        tracker.record(
            "join",
            {"left": len(left_incidents), "right": len(right_incidents)},
            blocked_pairs,
        )

        left_documents = load_selected_texts(
            "asrs",
            blocked_pairs,
            path_column="left_text_file",
            output_column="left_text",
        )[
            [
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
            ]
        ]
        tracker.record("scan", len(blocked_pairs), left_documents)

        left_plan = memory_dataset(
            f"{TASK_ID}-left-signatures", left_documents
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
                    "desc": "Distinctive measurements reported in the occurrence.",
                },
                {
                    "name": "left_action_sequence",
                    "type": list[str],
                    "desc": "The distinctive ordered sequence of actions.",
                },
                {
                    "name": "left_outcome",
                    "type": str,
                    "desc": "The reported outcome as a concise phrase.",
                },
            ],
            desc=(
                "Preserve both blocked-pair incident IDs and extract the "
                "normalized event signature from the left report."
            ),
            depends_on=[
                "left_incident_id",
                "right_incident_id",
                "left_text",
            ],
        )
        started = time.time()
        left_result = left_plan.run(config)
        left_signatures = result_frame(
            left_result,
            left_documents,
            [
                "left_primary_event_type",
                "left_measurements",
                "left_action_sequence",
                "left_outcome",
            ],
        )
        left_signatures["left_primary_event_type"] = left_signatures[
            "left_primary_event_type"
        ].map(lambda value: normalize_enum(value, EVENT_TYPES))
        tracker.record_semantic(
            "sem_map",
            len(left_documents),
            left_signatures,
            left_result,
            time.time() - started,
        )

        left_signatures = left_signatures.loc[
            left_signatures["left_primary_event_type"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(left_result), left_signatures)

        right_documents = load_selected_texts(
            "asrs",
            blocked_pairs,
            path_column="right_text_file",
            output_column="right_text",
        )[["left_incident_id", "right_incident_id", "right_text"]]
        tracker.record("scan", len(blocked_pairs), right_documents)

        right_plan = memory_dataset(
            f"{TASK_ID}-right-signatures", right_documents
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
                    "desc": "Distinctive measurements reported in the occurrence.",
                },
                {
                    "name": "right_action_sequence",
                    "type": list[str],
                    "desc": "The distinctive ordered sequence of actions.",
                },
                {
                    "name": "right_outcome",
                    "type": str,
                    "desc": "The reported outcome as a concise phrase.",
                },
            ],
            desc=(
                "Preserve both blocked-pair incident IDs and extract the same "
                "normalized event signature from the right report."
            ),
            depends_on=[
                "left_incident_id",
                "right_incident_id",
                "right_text",
            ],
        )
        started = time.time()
        right_result = right_plan.run(config)
        right_signatures = result_frame(
            right_result,
            right_documents,
            [
                "right_primary_event_type",
                "right_measurements",
                "right_action_sequence",
                "right_outcome",
            ],
        )
        right_signatures["right_primary_event_type"] = right_signatures[
            "right_primary_event_type"
        ].map(lambda value: normalize_enum(value, EVENT_TYPES))
        tracker.record_semantic(
            "sem_map",
            len(right_documents),
            right_signatures,
            right_result,
            time.time() - started,
        )

        right_signatures = right_signatures.loc[
            right_signatures["right_primary_event_type"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(right_result), right_signatures)

        left_dataset = memory_dataset(
            f"{TASK_ID}-left-match-input", left_signatures
        )
        right_dataset = memory_dataset(
            f"{TASK_ID}-right-match-input", right_signatures
        )
        match_plan = left_dataset.sem_join(
            right_dataset,
            condition=(
                "Compare only signature rows carrying the same left and right "
                "blocked-pair incident IDs. Match only when both narratives "
                "describe the same underlying occurrence, with agreement on a "
                "distinctive event progression and compatible measurements, "
                "actions, and outcome. Generic topical similarity is insufficient."
            ),
            depends_on=JOIN_INPUT_COLUMNS,
        )
        started = time.time()
        match_result = match_plan.run(config)
        matched = result_frame(match_result)
        if matched.empty:
            matched = pd.DataFrame(columns=MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(left_signatures), "right": len(right_signatures)},
            matched,
            match_result,
            time.time() - started,
        )

        matched_pairs = matched.loc[
            (matched["left_incident_id"] == matched["left_incident_id_right"])
            & (
                matched["right_incident_id"]
                == matched["right_incident_id_right"]
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(matched), matched_pairs)

        projected = pd.DataFrame(
            {
                "left_incident_id": matched_pairs["left_incident_id"],
                "right_incident_id": matched_pairs["right_incident_id"],
                "incident_date": matched_pairs["incident_date"],
                "shared_primary_event_type": matched_pairs[
                    "left_primary_event_type"
                ],
                "differing_structured_fields": matched_pairs.apply(
                    differing_fields, axis=1
                ),
            }
        )
        tracker.record("project", len(matched_pairs), projected)

        ordered = projected.sort_values(
            ["incident_date", "left_incident_id", "right_incident_id"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("sort", len(projected), ordered)

        limited = ordered.head(20).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited.copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
