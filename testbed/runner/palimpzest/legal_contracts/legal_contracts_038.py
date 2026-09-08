#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-038."""

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

TASK_ID = "legal_contracts-038"
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
                "Keep the document only if it is a non-disclosure agreement or "
                "an EX-10 material contract exhibit."
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
                    "name": "exhibit_family",
                    "type": str | None,
                    "desc": (
                        "Use NDA for an NDA, the applicable EX-xx label for a "
                        "material contract exhibit, and NDA|EX-xx when the "
                        "document has both statuses."
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
                {
                    "name": "term_months",
                    "type": float | None,
                    "desc": (
                        "The finite confidentiality term normalized to months, "
                        "or null when unstated, perpetual, or unquantifiable."
                    ),
                },
                {
                    "name": "non_compete_months",
                    "type": float | None,
                    "desc": (
                        "The quantifiable non-compete period in months, or null "
                        "when absent or unquantifiable."
                    ),
                },
                {
                    "name": "customer_non_solicitation_months",
                    "type": float | None,
                    "desc": (
                        "The quantifiable customer-non-solicitation period in "
                        "months, or null when absent or unquantifiable."
                    ),
                },
                {
                    "name": "employee_non_solicitation_months",
                    "type": float | None,
                    "desc": (
                        "The quantifiable employee-non-solicitation period in "
                        "months, or null when absent or unquantifiable."
                    ),
                },
            ],
            desc=(
                "Extract exhibit family, governing law, finite confidentiality "
                "term, and all quantifiable restriction periods."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "exhibit_family",
            "governing_law",
            "term_months",
            "non_compete_months",
            "customer_non_solicitation_months",
            "employee_non_solicitation_months",
        ]
        extracted = result_frame(extraction_result, agreements, generated)
        extracted["exhibit_family"] = extracted["exhibit_family"].map(
            normalize_text
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        for column in generated[2:]:
            extracted[column] = extracted[column].map(normalize_number)
        tracker.record_semantic(
            "sem_map",
            len(agreements),
            extracted,
            extraction_result,
            time.time() - started,
        )

        projected = extracted.copy()
        projected["longest_restriction_months"] = projected[
            [
                "non_compete_months",
                "customer_non_solicitation_months",
                "employee_non_solicitation_months",
            ]
        ].max(axis=1, skipna=True)
        projected = projected[
            [
                "document_id",
                "exhibit_family",
                "governing_law",
                "term_months",
                "longest_restriction_months",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        qualifying = projected.loc[
            projected["governing_law"].notna()
            & projected["term_months"].notna()
            & projected["longest_restriction_months"].notna()
            & (
                projected["term_months"]
                == projected["longest_restriction_months"]
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(projected), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
