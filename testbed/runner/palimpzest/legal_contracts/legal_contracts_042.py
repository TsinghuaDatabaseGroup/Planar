#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-042."""

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

TASK_ID = "legal_contracts-042"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_term_years(value):
    value = normalize_scalar_value(value)
    normalized = str(value).strip().lower()
    if "perpetual" in normalized or "indefinite" in normalized:
        return "perpetual"
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def parse_optional_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure "
                "agreement."
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
                "Keep the agreement only if it expressly states a "
                "confidentiality term and governing law and contains at least "
                "one operative non-compete or employee non-solicitation clause."
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
                    "name": "term_duration_years",
                    "type": str,
                    "desc": (
                        "The confidentiality term as a numeric number of years "
                        "for a fixed duration, or exactly perpetual for an "
                        "indefinite or perpetual term."
                    ),
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": "The expressly stated governing-law jurisdiction.",
                },
                {
                    "name": "non_compete_present",
                    "type": bool | None,
                    "desc": (
                        "True if an operative non-compete clause is present; "
                        "false otherwise."
                    ),
                },
                {
                    "name": "non_solicitation_employees_present",
                    "type": bool | None,
                    "desc": (
                        "True if an operative employee non-solicitation clause "
                        "is present; false otherwise."
                    ),
                },
            ],
            desc=(
                "Extract the normalized confidentiality term, governing law, "
                "and both clause indicators."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "term_duration_years",
            "governing_law",
            "non_compete_present",
            "non_solicitation_employees_present",
        ]
        extracted = result_frame(extraction_result, qualifying, generated)
        extracted["term_duration_years"] = extracted[
            "term_duration_years"
        ].map(normalize_term_years)
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        for column in (
            "non_compete_present",
            "non_solicitation_employees_present",
        ):
            extracted[column] = extracted[column].map(parse_optional_bool)
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
