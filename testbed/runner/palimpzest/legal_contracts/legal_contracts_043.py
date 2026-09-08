#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-043."""

from __future__ import annotations

import ast
import json
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
    load_mixed_documents,
    memory_dataset,
    normalize_text_value,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-043"
DATASET = "contract-exhibit"
AGREEMENT_TYPES = ("mutual", "unilateral")


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_string_list(value) -> list[str]:
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return []
    parsed = value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value.strip())
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
    if pd.api.types.is_list_like(parsed) and not isinstance(parsed, dict):
        parsed = list(parsed)
    else:
        parsed = [parsed]
    return [
        normalized
        for item in parsed
        if (normalized := " ".join(str(item).strip().split()))
    ]


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        exhibit_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is an EX-10 material contract "
                "exhibit."
            ),
            depends_on=["text"],
        )
        started = time.time()
        exhibit_result = exhibit_plan.run(config)
        exhibits = result_frame(exhibit_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            exhibits,
            exhibit_result,
            time.time() - started,
        )

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", exhibits
        ).sem_filter(
            filter=(
                "Keep the exhibit only if it is also a mutual or unilateral "
                "non-disclosure agreement and explicitly states an effective date."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, exhibits)
        tracker.record_semantic(
            "sem_filter",
            len(exhibits),
            qualifying,
            condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {
                    "name": "document_name",
                    "type": str | None,
                    "desc": "The stated name or title of the agreement.",
                },
                {
                    "name": "parties",
                    "type": list[str],
                    "desc": "All expressly named agreement parties.",
                },
                {
                    "name": "agreement_type",
                    "type": str | None,
                    "desc": "Exactly mutual or unilateral.",
                },
                {
                    "name": "effective_date",
                    "type": str | None,
                    "desc": "The explicitly stated effective date as YYYY-MM-DD.",
                },
            ],
            desc=(
                "Extract the document name, parties, agreement type, and "
                "effective date."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["document_name", "parties", "agreement_type", "effective_date"]
        extracted = result_frame(extraction_result, qualifying, generated)
        extracted["document_name"] = extracted["document_name"].map(normalize_text)
        extracted["parties"] = extracted["parties"].map(normalize_string_list)
        extracted["agreement_type"] = extracted["agreement_type"].map(
            lambda value: normalize_enum(value, AGREEMENT_TYPES)
        )
        extracted["effective_date"] = extracted["effective_date"].map(normalize_text)
        extracted = extracted[generated].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(qualifying),
            extracted,
            extraction_result,
            time.time() - started,
        )

        answer = df_records(extracted)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
