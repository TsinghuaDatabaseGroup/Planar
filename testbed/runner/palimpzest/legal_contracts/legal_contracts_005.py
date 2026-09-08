#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-005."""

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
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-005"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value,
        null_markers={
            "", "none", "null", "n/a", "unknown", "unstated", "unquantifiable"
        },
    )


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
                "confidentiality-and-standstill agreement."
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
                    "name": "stated_duration",
                    "type": str | None,
                    "desc": (
                        "The original wording that states the confidentiality "
                        "duration, or null if no duration is stated."
                    ),
                },
                {
                    "name": "duration_months",
                    "type": float | None,
                    "desc": (
                        "The fixed confidentiality duration normalized to months; "
                        "null when unstated, perpetual, indefinite, or otherwise "
                        "unquantifiable."
                    ),
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The expressly stated governing-law jurisdiction, or "
                        "null when unstated."
                    ),
                },
            ],
            desc=(
                "Extract the original confidentiality-duration wording, the "
                "duration normalized to months, and the governing law."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            agreements,
            ["stated_duration", "duration_months", "governing_law"],
        )
        extracted["stated_duration"] = extracted["stated_duration"].map(
            normalize_text
        )
        extracted["duration_months"] = extracted["duration_months"].map(
            normalize_number
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        tracker.record_semantic(
            "sem_map",
            len(agreements),
            extracted,
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["duration_months"].notna()
            & (extracted["duration_months"] <= 12)
            & extracted["governing_law"].notna(),
            ["document_id", "stated_duration", "duration_months", "governing_law"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
