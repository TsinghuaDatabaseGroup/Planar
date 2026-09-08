#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-049."""

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
    load_mixed_documents,
    memory_dataset,
    normalize_scalar_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-049"
DATASET = "contract-exhibit"


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure "
                "agreement, unilateral non-disclosure agreement, or "
                "confidentiality-and-standstill agreement with a specified "
                "finite, non-perpetual confidentiality term."
            ),
            depends_on=["text"],
        )
        started = time.time()
        agreement_result = agreement_plan.run(config)
        agreements = result_frame(agreement_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            agreements,
            agreement_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", agreements
        ).sem_map(
            cols=[
                {
                    "name": "term_duration_years",
                    "type": float | None,
                    "desc": (
                        "The finite confidentiality term normalized to years."
                    ),
                }
            ],
            desc="Extract the finite confidentiality term normalized to years.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result, agreements, ["term_duration_years"]
        )
        extracted["term_duration_years"] = extracted[
            "term_duration_years"
        ].map(normalize_number)
        extracted = extracted[
            ["document_id", "term_duration_years"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(agreements),
            extracted,
            extraction_result,
            time.time() - started,
        )

        ordered = extracted.sort_values(
            ["term_duration_years", "document_id"],
            ascending=[False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(extracted), ordered)

        limited = ordered.head(1).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        rows = df_records(limited)
        answer = rows[0] if rows else {}

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
