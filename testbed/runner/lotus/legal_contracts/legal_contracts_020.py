#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-020."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    normalize_enum,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-020"
DIRECTIONS = ("mutual", "unilateral")


def main():
    setup(max_tokens=512, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all)",
            None,
            len(documents),
            output=documents,
        )

        short_documents = documents.loc[
            documents["word_count"] < 1500
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(word_count < 1500)",
            len(documents),
            len(short_documents),
            output=short_documents,
        )

        with tracker.step(
            "SEM_FILTER(standalone NDA or confidentiality agreement)",
            input_rows=len(short_documents),
        ) as step:
            standalone_agreements = short_documents.sem_filter(
                "The document {text} is itself a standalone non-disclosure or "
                "confidentiality agreement, rather than another kind of document "
                "that only mentions confidentiality."
            ).reset_index(drop=True)
            step.set_output(standalone_agreements)

        with tracker.step(
            "SEM_FILTER(governing law, perpetual duty, return/destroy, and relief)",
            input_rows=len(standalone_agreements),
        ) as step:
            qualifying = standalone_agreements.sem_filter(
                "The agreement {text} expressly states a governing law, makes the "
                "confidentiality obligation indefinite or perpetual, requires "
                "confidential materials to be returned or destroyed, and provides "
                "injunctive relief, equitable relief, or specific performance. All "
                "four conditions must be present."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(governing law and nondisclosure direction)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction"
                    ),
                    "direction": (
                        "exactly mutual if both sides owe the nondisclosure duty, "
                        "or unilateral if only one side owes it"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted["direction"] = extracted["direction"].map(
                lambda value: normalize_enum(value, DIRECTIONS)
            )
            extracted = extracted[
                ["document_id", "governing_law", "direction"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
