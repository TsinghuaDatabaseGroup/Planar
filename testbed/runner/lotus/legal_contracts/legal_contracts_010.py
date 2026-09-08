#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-010."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    parse_number,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-010"


def main():
    setup(max_tokens=1024, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(unilateral NDA with named parties, law, and term <= 2 years)",
            input_rows=len(documents),
        ) as step:
            qualifying = documents.sem_filter(
                "The document {text} is a unilateral non-disclosure agreement, names "
                "at least one disclosing party and at least one receiving party, "
                "expressly states a governing law, and has a fixed confidentiality "
                "term of no more than two years. All conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(parties, term in years, and governing law)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "disclosing_parties": (
                        "a list of all expressly named disclosing parties"
                    ),
                    "receiving_parties": (
                        "a list of all expressly named receiving parties"
                    ),
                    "term_duration_years": (
                        "the fixed confidentiality term normalized to years as a number"
                    ),
                    "governing_law": "the expressly stated governing-law jurisdiction",
                },
            )
            extracted["disclosing_parties"] = extracted[
                "disclosing_parties"
            ].map(parse_string_list)
            extracted["receiving_parties"] = extracted["receiving_parties"].map(
                parse_string_list
            )
            extracted["term_duration_years"] = extracted[
                "term_duration_years"
            ].map(parse_number)
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted = extracted[
                [
                    "document_id",
                    "disclosing_parties",
                    "receiving_parties",
                    "term_duration_years",
                    "governing_law",
                ]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
