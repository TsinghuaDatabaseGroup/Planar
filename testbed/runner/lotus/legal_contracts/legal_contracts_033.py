#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-033."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-033"


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
            "SEM_FILTER(mutual or unilateral NDA)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual or unilateral non-disclosure "
                "agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(no non-compete, non-solicitation, or standstill restriction)",
            input_rows=len(agreements),
        ) as step:
            unrestricted = agreements.sem_filter(
                "The agreement {text} contains no operative non-compete restriction, "
                "no customer, employee, or vendor non-solicitation restriction, and "
                "no operative standstill restriction. All absence conditions must "
                "hold."
            ).reset_index(drop=True)
            step.set_output(unrestricted)

        with tracker.step(
            "SEM_EXTRACT(governing law and finite confidentiality duration)",
            input_rows=len(unrestricted),
        ) as step:
            extracted = unrestricted.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                    "duration_months": (
                        "the finite confidentiality duration normalized to months as "
                        "an integer; null when unstated, perpetual, indefinite, or "
                        "otherwise unquantifiable"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            extracted["duration_months"] = extracted["duration_months"].map(
                parse_number
            )
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["governing_law"].notna()
            & extracted["duration_months"].notna()
            & (extracted["duration_months"] > 60),
            ["document_id", "governing_law", "duration_months"],
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(governing_law IS NOT NULL AND duration_months > 60)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
