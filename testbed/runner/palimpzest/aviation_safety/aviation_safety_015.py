#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-015."""

from __future__ import annotations

import ast
import json
import os
import re
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
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-015"
CONSEQUENCES = (
    "emergency_declaration",
    "diversion",
    "return",
    "holding",
    "continued_with_concern",
)
CONSEQUENCE_PRIORITY = {
    label: index for index, label in enumerate(CONSEQUENCES, start=1)
}


def normalize_numeric_list(value) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        items = []
    else:
        text = str(value).strip()
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
        items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [text]

    normalized = []
    seen = set()
    for item in items:
        text = " ".join(str(item).split())
        if not text or not re.search(r"\d", text) or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


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
                "mission",
                "result_summary",
            ],
        )
        tracker.record("scan", None, incidents)

        part121_passenger = incidents.loc[
            (incidents["far_part"] == "Part 121")
            & (incidents["mission"] == "Passenger")
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), part121_passenger)

        reports = load_selected_texts("asrs", part121_passenger)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record("scan", len(part121_passenger), reports)

        extraction_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "qualifies",
                    "type": bool,
                    "desc": (
                        "True only when a fuel quantity, reserve, calculation, "
                        "or dispatch-release decision directly affected the "
                        "flight's operation."
                    ),
                },
                {
                    "name": "fuel_values_normalized",
                    "type": list[str],
                    "desc": (
                        "A short list of explicit numeric fuel quantities, "
                        "reserves, or calculation values relevant to the "
                        "operational decision, each with its unit when stated; "
                        "otherwise an empty list."
                    ),
                },
                {
                    "name": "time_values_normalized",
                    "type": list[str],
                    "desc": (
                        "A separate short list of explicit numeric time values "
                        "relevant to the operational decision, each with its "
                        "unit when stated; otherwise an empty list."
                    ),
                },
                {
                    "name": "operational_consequence",
                    "type": str,
                    "desc": (
                        "The highest applicable consequence using this order: "
                        "emergency_declaration, diversion, return, holding, "
                        "continued_with_concern."
                    ),
                },
            ],
            desc=(
                "Extract one record when fuel information or a dispatch-release "
                "decision directly affected the operation. Keep fuel and time "
                "values separate. The result summary may establish an emergency, "
                "diversion, or return only when consistent with the narrative."
            ),
            depends_on=["incident_id", "result_summary", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result)
        extracted["qualifies"] = extracted["qualifies"].map(parse_bool)
        extracted["fuel_values_normalized"] = extracted[
            "fuel_values_normalized"
        ].map(normalize_numeric_list)
        extracted["time_values_normalized"] = extracted[
            "time_values_normalized"
        ].map(normalize_numeric_list)
        extracted["operational_consequence"] = extracted[
            "operational_consequence"
        ].map(lambda value: normalize_enum(value, CONSEQUENCES))
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            extraction_result,
            time.time() - started,
        )

        selected = extracted.loc[
            extracted["qualifies"]
            & extracted["operational_consequence"].notna(),
            [
                "incident_id",
                "fuel_values_normalized",
                "time_values_normalized",
                "operational_consequence",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), selected)

        projected = selected.copy()
        projected["consequence_priority"] = projected[
            "operational_consequence"
        ].map(CONSEQUENCE_PRIORITY)
        projected["has_numeric_value"] = (
            projected["fuel_values_normalized"].map(bool)
            | projected["time_values_normalized"].map(bool)
        )
        tracker.record("project", len(selected), projected)

        ordered = projected.sort_values(
            ["consequence_priority", "has_numeric_value", "incident_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        tracker.record("sort", len(projected), ordered)

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited[
            [
                "incident_id",
                "fuel_values_normalized",
                "time_values_normalized",
                "operational_consequence",
            ]
        ].copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
