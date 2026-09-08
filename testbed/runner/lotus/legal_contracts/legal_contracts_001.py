#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-001."""

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

TASK_ID = "legal_contracts-001"


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

        short_documents = documents.loc[
            documents["word_count"] < 200
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(word_count < 200)",
            len(documents),
            len(short_documents),
            output=short_documents,
        )

        with tracker.step(
            "SEM_FILTER(independent auditor consent for SEC filing)",
            input_rows=len(short_documents),
        ) as step:
            auditor_consents = short_documents.sem_filter(
                "The document {text} is an independent auditor consent letter "
                "that authorizes the use or incorporation of an auditor's report "
                "in an SEC filing. Exclude consents from legal counsel, engineers, "
                "petroleum-reserve experts, and other non-auditor professionals."
            ).reset_index(drop=True)
            step.set_output(auditor_consents)

        answer = int(len(auditor_consents))
        tracker.record(
            "GROUP_BY([], count(*) AS document_count)",
            len(auditor_consents),
            1,
            output={"document_count": answer},
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
