#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-006."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    parse_label_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-006"
EXCEPTIONS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
)


def main():
    setup(max_tokens=768, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all)",
            None,
            len(documents),
            output=documents,
        )

        with tracker.step(
            "SEM_FILTER(NDA with perpetual or indefinite confidentiality duty)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement whose confidentiality duty is perpetual or "
                "indefinite."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(governing law and missing confidentiality exceptions)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                    "missing_exceptions": (
                        "a list containing every missing operative exception among "
                        "public_information, prior_knowledge, independent_development, "
                        "third_party_receipt, legal_compulsion, and consent"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            extracted["missing_exceptions"] = extracted[
                "missing_exceptions"
            ].map(lambda value: parse_label_list(value, EXCEPTIONS))
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["governing_law"].notna()
            & (extracted["missing_exceptions"].map(len) == 2),
            ["document_id", "governing_law", "missing_exceptions"],
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(governing_law IS NOT NULL AND CARDINALITY(missing_exceptions) = 2)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
