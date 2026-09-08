#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-023."""

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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-023"
DATASET = "contract-exhibit"
EXCEPTIONS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
)


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown"}
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

        duration_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is itself a standalone "
                "non-disclosure or confidentiality agreement, contains a "
                "standstill provision with a determinate duration, and states "
                "a determinate duration for at least one customer- or "
                "employee-non-solicitation provision. Incidental references "
                "unrelated to the confidentiality duty do not count."
            ),
            depends_on=["text"],
        )
        started = time.time()
        duration_result = duration_plan.run(config)
        duration_candidates = result_frame(duration_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            duration_candidates,
            duration_result,
            time.time() - started,
        )

        carve_out_plan = memory_dataset(
            f"{TASK_ID}-carve-outs", duration_candidates
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains exactly five, not "
                "fewer and not all six, of these operative confidentiality "
                "carve-outs: public information, prior knowledge, independent "
                "development, lawful unrestricted third-party receipt, legally "
                "compelled disclosure, and disclosure authorized by the "
                "disclosing party."
            ),
            depends_on=["text"],
        )
        started = time.time()
        carve_out_result = carve_out_plan.run(config)
        five_exception_agreements = result_frame(
            carve_out_result, duration_candidates
        )
        tracker.record_semantic(
            "sem_filter",
            len(duration_candidates),
            five_exception_agreements,
            carve_out_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", five_exception_agreements
        ).sem_map(
            cols=[
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": "The expressly stated governing-law jurisdiction.",
                },
                {
                    "name": "missing_exception",
                    "type": str | None,
                    "desc": (
                        "The single missing carve-out, exactly one of "
                        "public_information, prior_knowledge, "
                        "independent_development, third_party_receipt, "
                        "legal_compulsion, or consent."
                    ),
                },
                {
                    "name": "standstill_months",
                    "type": float | None,
                    "desc": (
                        "The determinate standstill duration in months, using "
                        "30 days per month and 12 months per year."
                    ),
                },
                {
                    "name": "customer_non_solicitation_months",
                    "type": float | None,
                    "desc": (
                        "The determinate customer-non-solicitation duration in "
                        "months using 30 days per month and 12 months per year, "
                        "or null if no determinate duration is stated."
                    ),
                },
                {
                    "name": "employee_non_solicitation_months",
                    "type": float | None,
                    "desc": (
                        "The determinate employee-non-solicitation duration in "
                        "months using 30 days per month and 12 months per year, "
                        "or null if no determinate duration is stated."
                    ),
                },
            ],
            desc=(
                "Extract the governing law, sole missing carve-out, and the "
                "three determinate durations normalized to months."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            five_exception_agreements,
            [
                "governing_law",
                "missing_exception",
                "standstill_months",
                "customer_non_solicitation_months",
                "employee_non_solicitation_months",
            ],
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        extracted["missing_exception"] = extracted["missing_exception"].map(
            lambda value: normalize_enum(value, EXCEPTIONS)
        )
        duration_columns = [
            "standstill_months",
            "customer_non_solicitation_months",
            "employee_non_solicitation_months",
        ]
        for column in duration_columns:
            extracted[column] = extracted[column].map(normalize_number)
        tracker.record_semantic(
            "sem_map",
            len(five_exception_agreements),
            extracted,
            extraction_result,
            time.time() - started,
        )

        projected = extracted.copy()
        projected["longest_solicitation_months"] = projected[
            [
                "customer_non_solicitation_months",
                "employee_non_solicitation_months",
            ]
        ].max(axis=1, skipna=True)
        projected = projected[
            [
                "document_id",
                "governing_law",
                "missing_exception",
                "standstill_months",
                "longest_solicitation_months",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        qualifying = projected.loc[
            projected["standstill_months"].notna()
            & projected["longest_solicitation_months"].notna()
            & (
                projected["standstill_months"]
                <= projected["longest_solicitation_months"]
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(projected), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
