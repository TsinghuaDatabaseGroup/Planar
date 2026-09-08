#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-006."""

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
    parse_label_list,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-006"
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
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


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
                "confidentiality-and-standstill agreement whose confidentiality "
                "duty is perpetual or indefinite."
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
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The expressly stated governing-law jurisdiction, or "
                        "null when unstated."
                    ),
                },
                {
                    "name": "missing_exceptions",
                    "type": list[str],
                    "desc": (
                        "Every missing operative carve-out among "
                        "public_information, prior_knowledge, "
                        "independent_development, third_party_receipt, "
                        "legal_compulsion, and consent."
                    ),
                },
            ],
            desc=(
                "Extract the governing law and all missing standard "
                "confidentiality carve-outs."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            agreements,
            ["governing_law", "missing_exceptions"],
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        extracted["missing_exceptions"] = extracted[
            "missing_exceptions"
        ].map(lambda value: parse_label_list(value, EXCEPTIONS))
        tracker.record_semantic(
            "sem_map",
            len(agreements),
            extracted,
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["governing_law"].notna()
            & (extracted["missing_exceptions"].map(len) == 2),
            ["document_id", "governing_law", "missing_exceptions"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
