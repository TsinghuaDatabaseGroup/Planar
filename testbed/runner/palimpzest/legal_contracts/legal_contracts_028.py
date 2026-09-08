#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-028."""

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

TASK_ID = "legal_contracts-028"
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

        clause_plan = memory_dataset(
            f"{TASK_ID}-clauses", agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains operative standstill "
                "and employee non-solicitation restrictions and contains no "
                "operative non-compete restriction."
            ),
            depends_on=["text"],
        )
        started = time.time()
        clause_result = clause_plan.run(config)
        qualifying_clauses = result_frame(clause_result, agreements)
        tracker.record_semantic(
            "sem_filter",
            len(agreements),
            qualifying_clauses,
            clause_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying_clauses
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
                    "name": "standstill_months",
                    "type": float | None,
                    "desc": (
                        "The standstill period in months; multiply years by 12 "
                        "and convert a fixed-date interval as elapsed_days * 12 "
                        "/ 365 rounded to two decimals; null if unquantifiable."
                    ),
                },
                {
                    "name": "employee_non_solicitation_months",
                    "type": float | None,
                    "desc": (
                        "The employee-non-solicitation period in months; multiply "
                        "years by 12 and convert a fixed-date interval as "
                        "elapsed_days * 12 / 365 rounded to two decimals; null "
                        "if unquantifiable."
                    ),
                },
            ],
            desc=(
                "Extract the governing law and normalized standstill and "
                "employee-non-solicitation periods."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            qualifying_clauses,
            [
                "governing_law",
                "standstill_months",
                "employee_non_solicitation_months",
            ],
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        for column in ("standstill_months", "employee_non_solicitation_months"):
            extracted[column] = extracted[column].map(normalize_number)
        tracker.record_semantic(
            "sem_map",
            len(qualifying_clauses),
            extracted,
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["standstill_months"].notna()
            & extracted["employee_non_solicitation_months"].notna()
            & (
                extracted["standstill_months"]
                <= extracted["employee_non_solicitation_months"]
            ),
            [
                "document_id",
                "governing_law",
                "standstill_months",
                "employee_non_solicitation_months",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
