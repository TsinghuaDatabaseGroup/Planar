#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-030."""

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

TASK_ID = "legal_contracts-030"
EXCEPTIONS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
)


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
            "SEM_FILTER(confidentiality-and-standstill agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a confidentiality-and-standstill agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(governing law and exactly five carve-outs)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} expressly states a governing law and contains "
                "exactly five, not fewer or all six, of these operative "
                "confidentiality exceptions: publicly available information, prior "
                "knowledge, independent development, lawful unrestricted third-party "
                "receipt, legally compelled disclosure, and disclosure authorized "
                "by the disclosing party."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(governing law and sole missing exception)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": "the expressly stated governing-law jurisdiction",
                    "missing_exception": (
                        "the sole missing exception, exactly one of public_information, "
                        "prior_knowledge, independent_development, third_party_receipt, "
                        "legal_compulsion, or consent"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted["missing_exception"] = extracted["missing_exception"].map(
                lambda value: normalize_enum(value, EXCEPTIONS)
            )
            extracted = extracted[
                ["document_id", "governing_law", "missing_exception"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
