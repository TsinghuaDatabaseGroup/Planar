#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-030."""

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

TASK_ID = "legal_contracts-030"
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


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a "
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

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it expressly states a governing "
                "law and contains exactly five, not fewer and not all six, of "
                "these operative confidentiality carve-outs: publicly available "
                "information, prior knowledge, independent development, lawful "
                "unrestricted third-party receipt, legally compelled disclosure, "
                "and disclosure authorized by the disclosing party."
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
                    "name": "missing_exception",
                    "type": str | None,
                    "desc": (
                        "The sole missing carve-out, exactly one of "
                        "public_information, prior_knowledge, "
                        "independent_development, third_party_receipt, "
                        "legal_compulsion, or consent."
                    ),
                },
            ],
            desc="Extract the governing law and sole missing carve-out.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            qualifying,
            ["governing_law", "missing_exception"],
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        extracted["missing_exception"] = extracted["missing_exception"].map(
            lambda value: normalize_enum(value, EXCEPTIONS)
        )
        extracted = extracted[
            ["document_id", "governing_law", "missing_exception"]
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
