#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-021."""

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

TASK_ID = "legal_contracts-021"


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

        text_documents = documents.loc[
            documents["file_extension"] == ".txt"
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(file_extension = '.txt')",
            len(documents),
            len(text_documents),
            output=text_documents,
        )

        with tracker.step(
            "SEM_FILTER(standalone NDA or confidentiality agreement)",
            input_rows=len(text_documents),
        ) as step:
            standalone_agreements = text_documents.sem_filter(
                "The document {text} is itself a standalone non-disclosure or "
                "confidentiality agreement, rather than another kind of document "
                "that only mentions confidentiality."
            ).reset_index(drop=True)
            step.set_output(standalone_agreements)

        with tracker.step(
            "SEM_FILTER(governing law, six carve-outs, and no non-solicitation)",
            input_rows=len(standalone_agreements),
        ) as step:
            qualifying = standalone_agreements.sem_filter(
                "The agreement {text} expressly states a governing law; includes "
                "operative confidentiality exceptions for publicly available "
                "information, prior knowledge, independent development, lawful "
                "unrestricted third-party receipt, legally compelled disclosure, "
                "and disclosure authorized by the disclosing party; and contains "
                "neither customer nor employee non-solicitation language. Every "
                "listed condition must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(governing-law jurisdiction)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "every expressly named governing-law jurisdiction, using "
                        "full English jurisdiction names"
                    )
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
