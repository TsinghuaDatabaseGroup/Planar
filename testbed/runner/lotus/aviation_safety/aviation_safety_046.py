#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-046."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-046"
PRIMARY_EVENT_TYPES = (
    "equipment_failure",
    "airborne_conflict",
    "ground_conflict",
    "clearance_deviation",
    "weather_event",
    "maintenance_event",
    "cabin_event",
    "other",
)
BLOCK_KEYS = (
    "incident_date",
    "state_reference",
    "locale_reference_type",
    "primary_problem",
)
STRUCTURED_COMPARISON_FIELDS = (
    "flight_phase",
    "anomaly_summary",
    "result_summary",
)


def values_differ(left, right) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if left_missing and right_missing:
        return False
    if left_missing or right_missing:
        return True
    return str(left) != str(right)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
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
        left_incidents = load_table("asrs", "incidents.csv")[
            incident_columns
        ].copy()
        left_incidents = left_incidents.rename(
            columns={column: f"left_{column}" for column in incident_columns}
        )
        tracker.record(
            "SCAN_TABLE(incidents.csv AS left_incident)",
            None,
            len(left_incidents),
            output=left_incidents,
        )

        right_incidents = load_table("asrs", "incidents.csv")[
            incident_columns
        ].copy()
        right_incidents = right_incidents.rename(
            columns={column: f"right_{column}" for column in incident_columns}
        )
        tracker.record(
            "SCAN_TABLE(incidents.csv AS right_incident)",
            None,
            len(right_incidents),
            output=right_incidents,
        )

        left_complete = left_incidents.dropna(
            subset=[f"left_{column}" for column in BLOCK_KEYS]
        )
        right_complete = right_incidents.dropna(
            subset=[f"right_{column}" for column in BLOCK_KEYS]
        )
        blocked_pairs = left_complete.merge(
            right_complete,
            left_on=[f"left_{column}" for column in BLOCK_KEYS],
            right_on=[f"right_{column}" for column in BLOCK_KEYS],
            how="inner",
        )
        blocked_pairs = blocked_pairs[
            blocked_pairs["left_incident_id"]
            < blocked_pairs["right_incident_id"]
        ].reset_index(drop=True)
        tracker.record(
            "JOIN(left_incidents, right_incidents, IDs ordered and block keys equal)",
            {"left": len(left_incidents), "right": len(right_incidents)},
            len(blocked_pairs),
            output=blocked_pairs,
        )

        left_reports = load_selected_texts(
            "asrs",
            blocked_pairs,
            path_column="left_text_file",
            output_column="left_text",
        )[
            [
                "left_incident_id",
                "right_incident_id",
                "left_incident_date",
                "left_flight_phase",
                "right_flight_phase",
                "left_anomaly_summary",
                "right_anomaly_summary",
                "left_result_summary",
                "right_result_summary",
                "left_text",
            ]
        ]
        tracker.record(
            "SCAN_DOCS(selector=blocked_pairs.left_incident.text_file)",
            len(blocked_pairs),
            len(left_reports),
            output=left_reports,
        )

        with tracker.step(
            "SEM_EXTRACT(left occurrence signature)",
            input_rows=len(left_reports),
        ) as step:
            if left_reports.empty:
                left_signatures = pd.DataFrame(
                    columns=[
                        "left_pair_left_incident_id",
                        "left_pair_right_incident_id",
                        "source_incident_date",
                        "source_left_flight_phase",
                        "source_right_flight_phase",
                        "source_left_anomaly_summary",
                        "source_right_anomaly_summary",
                        "source_left_result_summary",
                        "source_right_result_summary",
                        "left_primary_event_type",
                        "left_measurements",
                        "left_action_sequence",
                        "left_outcome",
                    ]
                )
            else:
                extracted_left = left_reports.sem_extract(
                    input_cols=["left_text"],
                    output_cols={
                        "left_primary_event_type": (
                            "the normalized primary event type, exactly one of "
                            "equipment_failure, airborne_conflict, ground_conflict, "
                            "clearance_deviation, weather_event, maintenance_event, "
                            "cabin_event, other"
                        ),
                        "left_measurements": (
                            "a JSON array of distinctive reported measurements or "
                            "quantities, preserving units; use an empty array if none"
                        ),
                        "left_action_sequence": (
                            "a JSON array of concise actions in their reported order"
                        ),
                        "left_outcome": (
                            "the final operational outcome as a concise factual phrase"
                        ),
                    },
                )
                extracted_left["left_primary_event_type"] = extracted_left[
                    "left_primary_event_type"
                ].map(lambda value: normalize_enum(value, PRIMARY_EVENT_TYPES))
                extracted_left["left_measurements"] = extracted_left[
                    "left_measurements"
                ].map(parse_string_list)
                extracted_left["left_action_sequence"] = extracted_left[
                    "left_action_sequence"
                ].map(parse_string_list)
                extracted_left["left_outcome"] = extracted_left[
                    "left_outcome"
                ].map(lambda value: clean_text(value, default=""))
                extracted_left = extracted_left.loc[
                    extracted_left["left_primary_event_type"].notna()
                    & extracted_left["left_outcome"].ne("")
                ].copy()
                left_signatures = extracted_left.rename(
                    columns={
                        "left_incident_id": "left_pair_left_incident_id",
                        "right_incident_id": "left_pair_right_incident_id",
                        "left_incident_date": "source_incident_date",
                        "left_flight_phase": "source_left_flight_phase",
                        "right_flight_phase": "source_right_flight_phase",
                        "left_anomaly_summary": "source_left_anomaly_summary",
                        "right_anomaly_summary": "source_right_anomaly_summary",
                        "left_result_summary": "source_left_result_summary",
                        "right_result_summary": "source_right_result_summary",
                    }
                )[
                    [
                        "left_pair_left_incident_id",
                        "left_pair_right_incident_id",
                        "source_incident_date",
                        "source_left_flight_phase",
                        "source_right_flight_phase",
                        "source_left_anomaly_summary",
                        "source_right_anomaly_summary",
                        "source_left_result_summary",
                        "source_right_result_summary",
                        "left_primary_event_type",
                        "left_measurements",
                        "left_action_sequence",
                        "left_outcome",
                    ]
                ].reset_index(drop=True)
            step.set_output(left_signatures)

        right_reports = load_selected_texts(
            "asrs",
            blocked_pairs,
            path_column="right_text_file",
            output_column="right_text",
        )[["left_incident_id", "right_incident_id", "right_text"]]
        tracker.record(
            "SCAN_DOCS(selector=blocked_pairs.right_incident.text_file)",
            len(blocked_pairs),
            len(right_reports),
            output=right_reports,
        )

        with tracker.step(
            "SEM_EXTRACT(right occurrence signature)",
            input_rows=len(right_reports),
        ) as step:
            if right_reports.empty:
                right_signatures = pd.DataFrame(
                    columns=[
                        "right_pair_left_incident_id",
                        "right_pair_right_incident_id",
                        "right_primary_event_type",
                        "right_measurements",
                        "right_action_sequence",
                        "right_outcome",
                    ]
                )
            else:
                extracted_right = right_reports.sem_extract(
                    input_cols=["right_text"],
                    output_cols={
                        "right_primary_event_type": (
                            "the normalized primary event type, exactly one of "
                            "equipment_failure, airborne_conflict, ground_conflict, "
                            "clearance_deviation, weather_event, maintenance_event, "
                            "cabin_event, other"
                        ),
                        "right_measurements": (
                            "a JSON array of distinctive reported measurements or "
                            "quantities, preserving units; use an empty array if none"
                        ),
                        "right_action_sequence": (
                            "a JSON array of concise actions in their reported order"
                        ),
                        "right_outcome": (
                            "the final operational outcome as a concise factual phrase"
                        ),
                    },
                )
                extracted_right["right_primary_event_type"] = extracted_right[
                    "right_primary_event_type"
                ].map(lambda value: normalize_enum(value, PRIMARY_EVENT_TYPES))
                extracted_right["right_measurements"] = extracted_right[
                    "right_measurements"
                ].map(parse_string_list)
                extracted_right["right_action_sequence"] = extracted_right[
                    "right_action_sequence"
                ].map(parse_string_list)
                extracted_right["right_outcome"] = extracted_right[
                    "right_outcome"
                ].map(lambda value: clean_text(value, default=""))
                extracted_right = extracted_right.loc[
                    extracted_right["right_primary_event_type"].notna()
                    & extracted_right["right_outcome"].ne("")
                ].copy()
                right_signatures = extracted_right.rename(
                    columns={
                        "left_incident_id": "right_pair_left_incident_id",
                        "right_incident_id": "right_pair_right_incident_id",
                    }
                )[
                    [
                        "right_pair_left_incident_id",
                        "right_pair_right_incident_id",
                        "right_primary_event_type",
                        "right_measurements",
                        "right_action_sequence",
                        "right_outcome",
                    ]
                ].reset_index(drop=True)
            step.set_output(right_signatures)

        left_bindings = left_signatures.copy()
        left_bindings["left_signature_binding"] = left_bindings.apply(
            lambda row: (
                f"pair_ids=({row['left_pair_left_incident_id']}, "
                f"{row['left_pair_right_incident_id']}); "
                f"primary_event_type={row['left_primary_event_type']}; "
                f"measurements={row['left_measurements']}; "
                f"action_sequence={row['left_action_sequence']}; "
                f"outcome={row['left_outcome']}"
            ),
            axis=1,
        )
        tracker.record(
            "CODE_MAP(bind left occurrence signature)",
            len(left_signatures),
            len(left_bindings),
            output=left_bindings,
        )

        right_bindings = right_signatures.copy()
        right_bindings["right_signature_binding"] = right_bindings.apply(
            lambda row: (
                f"pair_ids=({row['right_pair_left_incident_id']}, "
                f"{row['right_pair_right_incident_id']}); "
                f"primary_event_type={row['right_primary_event_type']}; "
                f"measurements={row['right_measurements']}; "
                f"action_sequence={row['right_action_sequence']}; "
                f"outcome={row['right_outcome']}"
            ),
            axis=1,
        )
        tracker.record(
            "CODE_MAP(bind right occurrence signature)",
            len(right_signatures),
            len(right_bindings),
            output=right_bindings,
        )

        with tracker.step(
            "SEM_JOIN(signature rows describe same underlying occurrence)",
            input_rows={
                "left": len(left_bindings),
                "right": len(right_bindings),
            },
        ) as step:
            if left_bindings.empty or right_bindings.empty:
                matched_pairs = pd.DataFrame(
                    columns=[*left_bindings.columns, *right_bindings.columns]
                )
            else:
                matched_pairs = left_bindings.sem_join(
                    right_bindings,
                    "Compare left signature {left_signature_binding} with right "
                    "signature {right_signature_binding}. Match only when both rows "
                    "carry exactly the same ordered pair of blocked incident IDs and "
                    "the two narratives describe the same underlying occurrence. "
                    "Require the same primary event type, a distinctive compatible "
                    "event progression, and agreement on measurements, actions, and "
                    "outcome; generic topical similarity is insufficient."
                )
            step.set_output(matched_pairs)

        projected_records = []
        for row in matched_pairs.itertuples(index=False):
            differing_fields = [
                field
                for field in STRUCTURED_COMPARISON_FIELDS
                if values_differ(
                    getattr(row, f"source_left_{field}"),
                    getattr(row, f"source_right_{field}"),
                )
            ]
            projected_records.append(
                {
                    "left_incident_id": row.left_pair_left_incident_id,
                    "right_incident_id": row.left_pair_right_incident_id,
                    "incident_date": row.source_incident_date,
                    "shared_primary_event_type": row.left_primary_event_type,
                    "differing_structured_fields": differing_fields,
                }
            )
        projected = pd.DataFrame.from_records(
            projected_records,
            columns=[
                "left_incident_id",
                "right_incident_id",
                "incident_date",
                "shared_primary_event_type",
                "differing_structured_fields",
            ],
        )
        tracker.record(
            "PROJECT(pair, shared event type, differing structured fields)",
            len(matched_pairs),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            ["incident_date", "left_incident_id", "right_incident_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([incident_date DESC, incident IDs ASC])",
            len(projected),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(20).copy()
        tracker.record("LIMIT(20)", len(ordered), len(limited), output=limited)

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and likely duplicate-pair fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
