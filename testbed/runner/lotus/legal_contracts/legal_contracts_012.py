#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-012."""

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

TASK_ID = "legal_contracts-012"


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
            "SEM_FILTER(Section 302 certification with material-changes clause)",
            input_rows=len(documents),
        ) as step:
            certifications = documents.sem_filter(
                "The document {text} is a genuine Sarbanes-Oxley Section 302 "
                "certification and contains the material-changes disclosure clause "
                "for internal control over financial reporting."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_FILTER(generic clause with no specific control change)",
            input_rows=len(certifications),
        ) as step:
            generic_only = certifications.sem_filter(
                "The material-changes clause in {text} uses only generic certification "
                "language and describes no specific change to internal control over "
                "financial reporting."
            ).reset_index(drop=True)
            step.set_output(generic_only)

        answer = int(len(generic_only))
        tracker.record(
            "GROUP_BY([], COUNT(*) AS certification_count)",
            len(generic_only),
            1,
            output={"certification_count": answer},
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
