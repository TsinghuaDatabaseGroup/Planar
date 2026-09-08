#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-010."""

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
    normalize_scalar_value,
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-010"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


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

        condition_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a unilateral non-disclosure "
                "agreement, names at least one disclosing party and at least one "
                "receiving party, expressly states a governing law, and has a "
                "fixed confidentiality term of no more than two years."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            qualifying,
            condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {
                    "name": "disclosing_parties",
                    "type": list[str],
                    "desc": "All expressly named disclosing parties.",
                },
                {
                    "name": "receiving_parties",
                    "type": list[str],
                    "desc": "All expressly named receiving parties.",
                },
                {
                    "name": "term_duration_years",
                    "type": float | None,
                    "desc": "The fixed confidentiality term normalized to years.",
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": "The expressly stated governing-law jurisdiction.",
                },
            ],
            desc=(
                "Extract the disclosing and receiving parties, fixed term in "
                "years, and governing law."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "disclosing_parties",
            "receiving_parties",
            "term_duration_years",
            "governing_law",
        ]
        extracted = result_frame(extraction_result, qualifying, generated)
        for column in ("disclosing_parties", "receiving_parties"):
            extracted[column] = extracted[column].map(normalize_string_list)
        extracted["term_duration_years"] = extracted[
            "term_duration_years"
        ].map(normalize_number)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        extracted = extracted[
            ["document_id", *generated]
        ].reset_index(drop=True)
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
