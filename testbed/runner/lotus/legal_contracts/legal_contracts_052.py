#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-052."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    load_document_corpus,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-052"


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
            "SEM_FILTER(affiliate assignment without counterparty consent)",
            input_rows=len(agreements),
        ) as step:
            assignable = agreements.sem_filter(
                "An assignment provision in {text} permits assignment of the agreement "
                "or its contractual rights or obligations to a corporate affiliate "
                "without the counterparty's consent."
            ).reset_index(drop=True)
            step.set_output(assignable)

        with tracker.step(
            "SEM_FILTER(U.S. federal court forum)",
            input_rows=len(assignable),
        ) as step:
            federal_forum = assignable.sem_filter(
                "A forum-selection, venue, jurisdiction, or enforcement provision in "
                "{text} expressly identifies a United States federal court as a forum "
                "for disputes under the agreement."
            ).reset_index(drop=True)
            step.set_output(federal_forum)

        answer = int(len(federal_forum))
        tracker.record(
            "GROUP_BY([], COUNT(*) AS count)",
            len(federal_forum),
            1,
            output={"count": answer},
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
