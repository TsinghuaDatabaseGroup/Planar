#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-049."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-049"


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
            "SEM_FILTER(NDA with specified finite confidentiality term)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement with a specified finite, non-perpetual "
                "confidentiality term."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(confidentiality term in years)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "term_duration_years": (
                        "the finite confidentiality term normalized to years as a number"
                    )
                },
            )
            extracted["term_duration_years"] = extracted[
                "term_duration_years"
            ].map(parse_number)
            extracted = extracted[
                ["document_id", "term_duration_years"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        ordered = extracted.sort_values(
            ["term_duration_years", "document_id"],
            ascending=[False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(term_duration_years DESC, document_id ASC)",
            len(extracted),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(1).reset_index(drop=True)
        tracker.record(
            "LIMIT(1)",
            len(ordered),
            len(limited),
            output=limited,
        )
        rows = df_records(limited)
        answer = rows[0] if rows else {}

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
