#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-002."""

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

TASK_ID = "legal_contracts-002"


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
            "SEM_FILTER(SOX Section 302 or Section 906 certification)",
            input_rows=len(documents),
        ) as step:
            certifications = documents.sem_filter(
                "The document {text} is a Sarbanes-Oxley certification under "
                "Section 302 or Section 906."
            ).reset_index(drop=True)
            step.set_output(certifications)

        answer = int(len(certifications))
        tracker.record(
            "GROUP_BY([], count(*) AS document_count)",
            len(certifications),
            1,
            output={"document_count": answer},
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
