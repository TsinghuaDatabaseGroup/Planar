#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-015."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-015"
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
            "SEM_EXTRACT(direction and non-use restriction)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "non_disclosure_direction": (
                        "exactly mutual, unilateral, or unclear according to who owes "
                        "the nondisclosure duty"
                    ),
                    "has_non_use_restriction": (
                        "true if an explicit restriction on use of protected "
                        "information is present; false otherwise"
                    ),
                },
            )
            extracted["non_disclosure_direction"] = extracted[
                "non_disclosure_direction"
            ].map(lambda value: normalize_enum(value, DIRECTIONS))
            extracted["has_non_use_restriction"] = extracted[
                "has_non_use_restriction"
            ].map(parse_bool)
            step.set_output(extracted)

        grouped = (
            extracted.groupby("non_disclosure_direction", sort=False, dropna=False)
            .agg(
                agreement_count=("document_id", "size"),
                non_use_percentage=("has_non_use_restriction", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["non_use_percentage"] = (
                grouped["non_use_percentage"] * 100.0
            ).round(2)
        tracker.record(
            "GROUP_BY([direction], COUNT(*), ROUND(100 * AVG(non_use), 2))",
            len(extracted),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
