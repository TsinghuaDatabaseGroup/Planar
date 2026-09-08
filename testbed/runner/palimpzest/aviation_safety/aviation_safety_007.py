#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-007."""

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
    parse_bool,
    parse_label_list,
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-007"
DECISIONS = (
    "attempt_approach",
    "divert",
    "return",
    "hold",
    "leave_holding",
    "continue",
    "declare_minimum_fuel",
    "declare_fuel_emergency",
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
                "mission",
                "aircraft_operator",
                "result_summary",
            ],
        )
        tracker.record("scan", None, incidents)

        filtered = incidents.loc[
            (incidents["mission"] == "Passenger")
            & (incidents["aircraft_operator"] == "Air Carrier")
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
                        "True only when a stated fuel quantity, reserve, burn estimate, "
                        "or remaining-flight-time calculation directly shaped at least "
                        "one permitted operational decision."
                    ),
                },
                {
                    "name": "decision_labels",
                    "type": list[str],
                    "desc": (
                        "Every operational decision directly shaped by the fuel "
                        "information, selected only from attempt_approach, divert, "
                        "return, hold, leave_holding, continue, declare_minimum_fuel, "
                        "declare_fuel_emergency; return an empty list when none qualifies."
                    ),
                },
                {
                    "name": "has_numeric_fuel_value",
                    "type": bool,
                    "desc": (
                        "True only when the report states an explicit numeric fuel "
                        "quantity, reserve, or burn value relevant to the decision."
                    ),
                },
                {
                    "name": "has_numeric_time_value",
                    "type": bool,
                    "desc": (
                        "True only when the report states an explicit numeric time "
                        "value relevant to the fuel-based decision or remaining-flight-time calculation."
                    ),
                },
            ],
            desc=(
                "Identify each decision directly affected by fuel quantity, reserve, "
                "burn, or remaining-flight-time information and mark the two types of "
                "numeric evidence independently. The result summary may establish a "
                "diversion, return, or emergency only when the narrative confirms its "
                "relationship to the fuel information."
            ),
            depends_on=["result_summary", "text"],
        )
        started = time.time()
        semantic_result = semantic.run(config)
        raw = result_frame(semantic_result)
        raw["qualifies"] = raw["qualifies"].map(parse_bool)
        raw["decision_labels"] = raw["decision_labels"].map(
            lambda value: parse_label_list(value, DECISIONS)
        )
        raw["has_numeric_fuel_value"] = raw[
            "has_numeric_fuel_value"
        ].map(parse_bool)
        raw["has_numeric_time_value"] = raw[
            "has_numeric_time_value"
        ].map(parse_bool)

        rows = []
        for record in raw.itertuples(index=False):
            if not record.qualifies:
                continue
            for decision in record.decision_labels:
                rows.append(
                    {
                        "incident_id": record.incident_id,
                        "decision_label": decision,
                        "has_numeric_fuel_value": (
                            record.has_numeric_fuel_value
                        ),
                        "has_numeric_time_value": (
                            record.has_numeric_time_value
                        ),
                    }
                )
        extracted = pd.DataFrame.from_records(
            rows,
            columns=[
                "incident_id",
                "decision_label",
                "has_numeric_fuel_value",
                "has_numeric_time_value",
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
            extracted.groupby("decision_label", sort=False)
            .agg(
                incident_count=("incident_id", "nunique"),
                with_numeric_fuel_value_count=(
                    "incident_id",
                    lambda values: values[
                        extracted.loc[
                            values.index, "has_numeric_fuel_value"
                        ]
                    ].nunique(),
                ),
                with_numeric_time_value_count=(
                    "incident_id",
                    lambda values: values[
                        extracted.loc[
                            values.index, "has_numeric_time_value"
                        ]
                    ].nunique(),
                ),
            )
            .reset_index()
        )
        decision_order = {
            label: index for index, label in enumerate(DECISIONS)
        }
        grouped = grouped.sort_values(
            "decision_label",
            key=lambda values: values.map(decision_order),
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
