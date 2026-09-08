#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-045."""

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

TASK_ID = "legal_contracts-045"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def normalize_term_years(value):
    value = normalize_scalar_value(value)
    normalized = str(value).strip().lower()
    if "perpetual" in normalized or "indefinite" in normalized:
        return "perpetual"
    return normalize_number(value)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a non-disclosure agreement "
                "expressly governed by Delaware or New York law."
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

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it includes an operative standstill "
                "provision, recognizes all six standard confidentiality "
                "carve-outs for publicly available information, prior knowledge, "
                "independent development, lawful unrestricted third-party receipt, "
                "legally compelled disclosure, and disclosure authorized by the "
                "disclosing party, states a confidentiality term, and states a "
                "quantifiable standstill period."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, agreements)
        tracker.record_semantic(
            "sem_filter",
            len(agreements),
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
                    "name": "governing_state",
                    "type": str | None,
                    "desc": "The qualifying governing state written as its full name.",
                },
                {
                    "name": "term_duration_years",
                    "type": str,
                    "desc": (
                        "The confidentiality term as a number of years for a fixed "
                        "duration, or exactly perpetual for a perpetual or "
                        "indefinite term."
                    ),
                },
                {
                    "name": "standstill_period_years",
                    "type": float | None,
                    "desc": "The quantifiable standstill period normalized to years.",
                },
            ],
            desc=(
                "Extract the document name, governing state, confidentiality term, "
                "and standstill period."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "document_name",
            "governing_state",
            "term_duration_years",
            "standstill_period_years",
        ]
        extracted = result_frame(extraction_result, qualifying, generated)
        for column in ("document_name", "governing_state"):
            extracted[column] = extracted[column].map(normalize_text)
        extracted["term_duration_years"] = extracted[
            "term_duration_years"
        ].map(normalize_term_years)
        extracted["standstill_period_years"] = extracted[
            "standstill_period_years"
        ].map(normalize_number)
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
