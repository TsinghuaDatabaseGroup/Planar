#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-022."""

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

TASK_ID = "legal_contracts-022"
DATASET = "contract-exhibit"


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

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is itself a standalone "
                "non-disclosure or confidentiality agreement, rather than "
                "another kind of document that merely mentions confidentiality."
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
                "Keep the agreement only if it provides injunctive relief, "
                "equitable relief, or specific performance; states a finite "
                "confidentiality term of no more than 36 months; expressly "
                "states a governing law; contains all six operative carve-outs "
                "for publicly available information, prior knowledge, "
                "independent development, lawful unrestricted third-party "
                "receipt, legally compelled disclosure, and disclosure "
                "authorized by the disclosing party; and contains neither "
                "customer nor employee non-solicitation language."
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
                    "name": "governing_law",
                    "type": str | None,
                    "desc": "The expressly stated governing-law jurisdiction.",
                },
                {
                    "name": "term_months",
                    "type": float | None,
                    "desc": (
                        "The finite confidentiality term converted to months "
                        "and returned as a number."
                    ),
                },
            ],
            desc=(
                "Extract the governing law and finite confidentiality term "
                "normalized to months."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            qualifying,
            ["governing_law", "term_months"],
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        extracted["term_months"] = extracted["term_months"].map(
            normalize_number
        )
        extracted = extracted[
            ["document_id", "governing_law", "term_months"]
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
