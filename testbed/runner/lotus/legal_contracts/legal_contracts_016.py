#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-016."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, load_docs, save_output, setup  # noqa: E402


TASK_ID = "legal_contracts-016"


def main():
    setup(max_tokens=256)
    tracker = StepTracker()

    with Timer() as timer:
        documents = load_docs("contract-exhibit").rename(
            columns={"doc_id": "contract_id", "contents": "text"}
        )
        documents["doc_format"] = documents["contract_id"].map(
            lambda value: os.path.splitext(str(value))[1].lstrip(".").lower()
        )
        documents = documents[["contract_id", "doc_format", "text"]]
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all)",
            None,
            len(documents),
            output=documents,
        )

        html_documents = documents[documents["doc_format"] == "htm"].copy()
        tracker.record(
            "FILTER(doc_format='htm')",
            len(documents),
            len(html_documents),
            output=html_documents,
        )

        with tracker.step(
            "SEM_FILTER(substantive NDA or confidentiality document)",
            input_rows=len(html_documents),
        ) as step:
            qualifying = html_documents.sem_filter(
                "Keep this document when {text} substantively functions as a "
                "nondisclosure or confidentiality document, including when it "
                "also serves another SEC exhibit purpose. Exclude a document "
                "that merely mentions confidentiality without imposing a "
                "substantive confidentiality regime."
            )
            step.set_output(qualifying)

        answer = int(qualifying["contract_id"].nunique())
        tracker.record(
            "GROUP_BY([], COUNT_DISTINCT(contract_id))",
            len(qualifying),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
