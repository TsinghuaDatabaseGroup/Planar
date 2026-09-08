#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-026."""

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

TASK_ID = "legal_contracts-026"


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
            documents["word_count"] < 750
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(word_count < 750)",
            len(documents),
            len(short_documents),
            output=short_documents,
        )

        with tracker.step(
            "SEM_FILTER(non-disclosure agreement)",
            input_rows=len(short_documents),
        ) as step:
            agreements = short_documents.sem_filter(
                "The document {text} is a non-disclosure agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(law, IP ownership, and no residual-information right)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} expressly states a governing law, includes "
                "an intellectual-property ownership clause, and does not grant a "
                "residual-information right permitting use of confidential "
                "information retained in unaided memory. All conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(governing law)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": "the expressly stated governing-law jurisdiction"
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted = extracted[
                ["document_id", "governing_law"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
