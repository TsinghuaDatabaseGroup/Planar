#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-033."""

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

TASK_ID = "legal_contracts-033"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
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
                "Keep the document only if it is a mutual or unilateral "
                "non-disclosure agreement."
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

        restriction_plan = memory_dataset(
            f"{TASK_ID}-restrictions", agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains no operative "
                "non-compete restriction, no customer, employee, or vendor "
                "non-solicitation restriction, and no operative standstill "
                "restriction. All absence conditions must hold."
            ),
            depends_on=["text"],
        )
        started = time.time()
        restriction_result = restriction_plan.run(config)
        unrestricted = result_frame(restriction_result, agreements)
        tracker.record_semantic(
            "sem_filter",
            len(agreements),
            unrestricted,
            restriction_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", unrestricted
        ).sem_map(
            cols=[
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The expressly stated governing-law jurisdiction, or "
                        "null when unstated."
                    ),
                },
                {
                    "name": "duration_months",
                    "type": int | None,
                    "desc": (
                        "The finite confidentiality duration normalized to whole "
                        "months; null when unstated, perpetual, indefinite, or "
                        "otherwise unquantifiable."
                    ),
                },
            ],
            desc=(
                "Extract the governing law and finite confidentiality duration "
                "normalized to months."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            unrestricted,
            ["governing_law", "duration_months"],
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        extracted["duration_months"] = extracted["duration_months"].map(
            normalize_number
        )
        tracker.record_semantic(
            "sem_map",
            len(unrestricted),
            extracted,
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["governing_law"].notna()
            & extracted["duration_months"].notna()
            & (extracted["duration_months"] > 60),
            ["document_id", "governing_law", "duration_months"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
