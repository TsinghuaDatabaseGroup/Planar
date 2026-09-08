#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-024."""

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

TASK_ID = "legal_contracts-024"
EXCEPTIONS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
)
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

        with tracker.step(
            "SEM_FILTER(standalone agreement governed by California, Delaware, or New York)",
            input_rows=len(documents),
        ) as step:
            governed_agreements = documents.sem_filter(
                "The document {text} is itself a standalone non-disclosure or "
                "confidentiality agreement and is expressly governed by the law "
                "of California, Delaware, or New York."
            ).reset_index(drop=True)
            step.set_output(governed_agreements)

        with tracker.step(
            "SEM_FILTER(return/destroy, relief, and exactly five carve-outs)",
            input_rows=len(governed_agreements),
        ) as step:
            qualifying = governed_agreements.sem_filter(
                "The agreement {text} requires confidential materials to be returned "
                "or destroyed, provides injunctive relief, equitable relief, or "
                "specific performance, and contains exactly five of these six "
                "operative confidentiality exceptions: public information, prior "
                "knowledge, independent development, lawful unrestricted third-party "
                "receipt, legally compelled disclosure, and disclosure authorized "
                "by the disclosing party. All conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(governing state, missing carve-out, and direction)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_state": (
                        "the qualifying governing state, written as the full state name"
                    ),
                    "missing_exception": (
                        "the sole missing carve-out, exactly one of public_information, "
                        "prior_knowledge, independent_development, third_party_receipt, "
                        "legal_compulsion, or consent"
                    ),
                    "direction": (
                        "exactly mutual if both sides owe the nondisclosure duty, "
                        "or unilateral if only one side owes it"
                    ),
                },
            )
            extracted["governing_state"] = extracted["governing_state"].map(clean_text)
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
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
