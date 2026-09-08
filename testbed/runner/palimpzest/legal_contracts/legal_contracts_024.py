#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-024."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_mixed_documents,
    memory_dataset,
    normalize_text_value,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-024"
DATASET = "contract-exhibit"
EXCEPTIONS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
)
DIRECTIONS = ("mutual", "unilateral")


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown"}
    )


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        governed_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is itself a standalone "
                "non-disclosure or confidentiality agreement and is expressly "
                "governed by the law of California, Delaware, or New York."
            ),
            depends_on=["text"],
        )
        started = time.time()
        governed_result = governed_plan.run(config)
        governed_agreements = result_frame(governed_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            governed_agreements,
            governed_result,
            time.time() - started,
        )

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", governed_agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it requires confidential materials "
                "to be returned or destroyed, provides injunctive relief, "
                "equitable relief, or specific performance, and contains exactly "
                "five of these six operative confidentiality carve-outs: public "
                "information, prior knowledge, independent development, lawful "
                "unrestricted third-party receipt, legally compelled disclosure, "
                "and disclosure authorized by the disclosing party."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, governed_agreements)
        tracker.record_semantic(
            "sem_filter",
            len(governed_agreements),
            qualifying,
            condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {
                    "name": "governing_state",
                    "type": str | None,
                    "desc": (
                        "The qualifying governing state written as its full "
                        "English state name."
                    ),
                },
                {
                    "name": "missing_exception",
                    "type": str | None,
                    "desc": (
                        "The sole missing carve-out, exactly one of "
                        "public_information, prior_knowledge, "
                        "independent_development, third_party_receipt, "
                        "legal_compulsion, or consent."
                    ),
                },
                {
                    "name": "direction",
                    "type": str | None,
                    "desc": (
                        "Exactly mutual if both sides owe the nondisclosure duty, "
                        "or unilateral if only one side owes it."
                    ),
                },
            ],
            desc=(
                "Extract the qualifying governing state, sole missing carve-out, "
                "and nondisclosure direction."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            qualifying,
            ["governing_state", "missing_exception", "direction"],
        )
        extracted["governing_state"] = extracted["governing_state"].map(
            normalize_text
        )
        extracted["missing_exception"] = extracted["missing_exception"].map(
            lambda value: normalize_enum(value, EXCEPTIONS)
        )
        extracted["direction"] = extracted["direction"].map(
            lambda value: normalize_enum(value, DIRECTIONS)
        )
        extracted = extracted[
            [
                "document_id",
                "governing_state",
                "missing_exception",
                "direction",
            ]
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
