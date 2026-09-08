#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-064."""

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

TASK_ID = "legal_contracts-064"


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
            "SEM_FILTER(NDA with all six confidentiality exceptions)",
            input_rows=len(documents),
        ) as step:
            complete_exceptions = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement and contains all six operative confidentiality "
                "exceptions for public information, prior knowledge, independent "
                "development, lawful unrestricted third-party receipt, legal "
                "compulsion, and disclosure with consent."
            ).reset_index(drop=True)
            step.set_output(complete_exceptions)

        with tracker.step(
            "SEM_FILTER(injunctive relief, IP ownership, and survival clause)",
            input_rows=len(complete_exceptions),
        ) as step:
            complete_profiles = complete_exceptions.sem_filter(
                "The agreement {text} also contains injunctive relief, an "
                "intellectual-property ownership clause, and a clause stating that "
                "confidentiality obligations survive termination. All three must be "
                "present."
            ).reset_index(drop=True)
            step.set_output(complete_profiles)

        projected = complete_profiles[["document_id", "word_count"]].reset_index(
            drop=True
        )
        tracker.record(
            "PROJECT(document_id, WORD_COUNT(document_text) AS word_count)",
            len(complete_profiles),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            ["word_count", "document_id"],
            ascending=[True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(word_count ASC, document_id ASC)",
            len(projected),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(15).reset_index(drop=True)
        tracker.record(
            "LIMIT(15)",
            len(ordered),
            len(limited),
            output=limited,
        )
        answer = df_records(limited)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
