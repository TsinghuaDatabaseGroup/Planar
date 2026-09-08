#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-047."""

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
    parse_optional_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-047"
TERM_BUCKETS = ("fixed_months", "perpetual")


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
            "SEM_FILTER(NDA or confidentiality agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a non-disclosure or confidentiality agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(law, long or perpetual term, and residual-information clause)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} expressly states a governing law, has a fixed "
                "confidentiality term of at least 60 months or a perpetual or "
                "indefinite term, and contains a residual-information clause. All "
                "conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(law, term bucket, and archival-copy retention)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": "the expressly stated governing-law jurisdiction",
                    "term_bucket": (
                        "exactly fixed_months for a qualifying fixed term, or perpetual "
                        "for a perpetual or indefinite term"
                    ),
                    "allows_one_copy_retention": (
                        "true if the agreement permits retention of one archival copy "
                        "after return or destruction, otherwise false"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted["term_bucket"] = extracted["term_bucket"].map(
                lambda value: normalize_enum(value, TERM_BUCKETS)
            )
            extracted["allows_one_copy_retention"] = extracted[
                "allows_one_copy_retention"
            ].map(parse_optional_bool)
            extracted = extracted[
                [
                    "document_id",
                    "governing_law",
                    "term_bucket",
                    "allows_one_copy_retention",
                ]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
