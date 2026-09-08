#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-022."""

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
    save_output,
    setup,
)

TASK_ID = "legal_contracts-022"


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
            "SEM_FILTER(standalone NDA or confidentiality agreement)",
            input_rows=len(documents),
        ) as step:
            standalone_agreements = documents.sem_filter(
                "The document {text} is itself a standalone non-disclosure or "
                "confidentiality agreement, rather than another kind of document "
                "that only mentions confidentiality."
            ).reset_index(drop=True)
            step.set_output(standalone_agreements)

        with tracker.step(
            "SEM_FILTER(relief, finite term, law, six carve-outs, no non-solicitation)",
            input_rows=len(standalone_agreements),
        ) as step:
            qualifying = standalone_agreements.sem_filter(
                "The agreement {text} provides injunctive relief, equitable relief, "
                "or specific performance; states a finite confidentiality term of "
                "no more than 36 months; expressly states a governing law; includes "
                "operative exceptions for publicly available information, prior "
                "knowledge, independent development, lawful unrestricted third-party "
                "receipt, legally compelled disclosure, and disclosure authorized "
                "by the disclosing party; and contains neither customer nor employee "
                "non-solicitation language. Every condition must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(governing law and confidentiality term in months)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction"
                    ),
                    "term_months": (
                        "the finite confidentiality term converted to months, as a "
                        "number"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted["term_months"] = extracted["term_months"].map(parse_number)
            extracted = extracted[
                ["document_id", "governing_law", "term_months"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
