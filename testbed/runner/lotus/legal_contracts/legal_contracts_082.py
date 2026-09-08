#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-082."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-082"


def main():
    setup(max_tokens=128, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(NDA or confidentiality-and-standstill agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(perpetual or indefinite confidentiality obligation)",
            input_rows=len(agreements),
        ) as step:
            perpetual = agreements.sem_filter(
                "The confidentiality obligation in {text} is expressly perpetual or "
                "indefinite."
            ).reset_index(drop=True)
            step.set_output(perpetual)

        with tracker.step(
            "SEM_FILTER(no independent-development exception)",
            input_rows=len(perpetual),
        ) as step:
            no_independent_exception = perpetual.sem_filter(
                "The agreement {text} contains no operative confidentiality exception "
                "for independently developed information."
            ).reset_index(drop=True)
            step.set_output(no_independent_exception)

        with tracker.step(
            "SEM_FILTER(no severability clause)",
            input_rows=len(no_independent_exception),
        ) as step:
            no_severability = no_independent_exception.sem_filter(
                "The agreement {text} contains no severability clause."
            ).reset_index(drop=True)
            step.set_output(no_severability)

        projected = no_severability[["document_id"]].reset_index(drop=True)
        tracker.record(
            "PROJECT(document_id)",
            len(no_severability),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
