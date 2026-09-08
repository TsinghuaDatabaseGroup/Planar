#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-048."""

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

TASK_ID = "legal_contracts-048"
DATASET = "contract-exhibit"
DIRECTIONS = ("mutual", "unilateral")


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a non-disclosure or "
                "confidentiality agreement."
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
                "Keep the agreement only if it states a perpetual or indefinite "
                "confidentiality term and a governing law but contains no express "
                "clause stating that confidentiality duties survive termination."
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
                    "name": "non_disclosure_direction",
                    "type": str | None,
                    "desc": (
                        "Exactly mutual if both sides owe the nondisclosure duty, "
                        "or unilateral if only one side owes it."
                    ),
                },
            ],
            desc="Extract the governing law and nondisclosure direction.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["governing_law", "non_disclosure_direction"]
        extracted = result_frame(extraction_result, qualifying, generated)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        extracted["non_disclosure_direction"] = extracted[
            "non_disclosure_direction"
        ].map(lambda value: normalize_enum(value, DIRECTIONS))
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
