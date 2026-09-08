#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-043."""

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
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-043"
AGREEMENT_TYPES = ("mutual", "unilateral")


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
            "SEM_FILTER(EX-10 material contract exhibit)",
            input_rows=len(documents),
        ) as step:
            exhibits = documents.sem_filter(
                "The document {text} is an EX-10 material contract exhibit."
            ).reset_index(drop=True)
            step.set_output(exhibits)

        with tracker.step(
            "SEM_FILTER(mutual or unilateral NDA with explicit effective date)",
            input_rows=len(exhibits),
        ) as step:
            qualifying = exhibits.sem_filter(
                "The exhibit {text} is also a mutual or unilateral non-disclosure "
                "agreement and explicitly states an effective date."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(name, parties, type, and effective date)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "document_name": "the stated name or title of the agreement",
                    "parties": "a list of the expressly named agreement parties",
                    "agreement_type": "exactly mutual or unilateral",
                    "effective_date": (
                        "the explicitly stated effective date formatted as YYYY-MM-DD"
                    ),
                },
            )
            extracted["document_name"] = extracted["document_name"].map(clean_text)
            extracted["parties"] = extracted["parties"].map(parse_string_list)
            extracted["agreement_type"] = extracted["agreement_type"].map(
                lambda value: normalize_enum(value, AGREEMENT_TYPES)
            )
            extracted["effective_date"] = extracted["effective_date"].map(clean_text)
            extracted = extracted[
                ["document_name", "parties", "agreement_type", "effective_date"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
