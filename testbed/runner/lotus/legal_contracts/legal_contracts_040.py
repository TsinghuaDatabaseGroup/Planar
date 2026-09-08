#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-040."""

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

TASK_ID = "legal_contracts-040"


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
            "SEM_FILTER(mutual non-disclosure agreement)",
            input_rows=len(documents),
        ) as step:
            mutual_agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement."
            ).reset_index(drop=True)
            step.set_output(mutual_agreements)

        with tracker.step(
            "SEM_FILTER(confidential information includes source code or software)",
            input_rows=len(mutual_agreements),
        ) as step:
            software_agreements = mutual_agreements.sem_filter(
                "The definition of confidential information in {text} explicitly "
                "includes source code or software."
            ).reset_index(drop=True)
            step.set_output(software_agreements)

        with tracker.step(
            "SEM_FILTER(archival-copy retention after termination)",
            input_rows=len(software_agreements),
        ) as step:
            qualifying = software_agreements.sem_filter(
                "The agreement {text} permits the recipient to retain at least one "
                "archival copy of materials after termination."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        answer = "Yes" if len(qualifying) > 0 else "No"
        tracker.record(
            "GROUP_BY([], CASE WHEN COUNT(*) > 0 THEN 'Yes' ELSE 'No' END)",
            len(qualifying),
            1,
            output={"answer": answer},
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
