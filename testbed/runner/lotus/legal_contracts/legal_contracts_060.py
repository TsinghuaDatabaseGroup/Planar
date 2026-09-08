#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-060."""

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

TASK_ID = "legal_contracts-060"
DIRECTIONS = ("mutual", "unilateral", "unclear")


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
            "SEM_FILTER(unaided-memory residual right and governing law)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} contains a residual-information clause that "
                "permits use of information retained in unaided memory and expressly "
                "states a governing law. Both conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(nondisclosure direction and governing law)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "non_disclosure_direction": (
                        "exactly mutual, unilateral, or unclear according to who owes "
                        "the nondisclosure duty"
                    ),
                    "governing_law": "the expressly stated governing-law jurisdiction",
                },
            )
            extracted["non_disclosure_direction"] = extracted[
                "non_disclosure_direction"
            ].map(lambda value: normalize_enum(value, DIRECTIONS))
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted = extracted[
                ["document_id", "non_disclosure_direction", "governing_law"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
