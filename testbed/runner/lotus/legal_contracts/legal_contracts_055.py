#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-055."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-055"


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
            "SEM_FILTER(residual-information clause)",
            input_rows=len(documents),
        ) as step:
            residual_agreements = documents.sem_filter(
                "The agreement {text} contains a residual-information clause."
            ).reset_index(drop=True)
            step.set_output(residual_agreements)

        with tracker.step(
            "SEM_FILTER(governing law and IP or work-product ownership clause)",
            input_rows=len(residual_agreements),
        ) as step:
            qualifying = residual_agreements.sem_filter(
                "The agreement {text} expressly states a governing law and contains "
                "an intellectual-property or work-product ownership clause."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(exhibit type, agreement type, and governing law)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "exhibit_type": (
                        "the exhibit type; use NDA for an NDA, the applicable EX-xx "
                        "label for an exhibit, and NDA|EX-xx for a dual-status document"
                    ),
                    "agreement_type": "a concise stated agreement type",
                    "governing_law": "the expressly stated governing-law jurisdiction",
                },
            )
            for column in ("exhibit_type", "agreement_type", "governing_law"):
                extracted[column] = extracted[column].map(clean_text)
            extracted = extracted[
                ["document_id", "exhibit_type", "agreement_type", "governing_law"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
