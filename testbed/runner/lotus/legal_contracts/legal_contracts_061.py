#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-061."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_enum,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-061"
AGREEMENT_TYPES = (
    "mutual_nda",
    "unilateral_nda",
    "confidentiality_standstill",
    "not_in_scope",
)
DIRECTIONS = ("mutual", "unilateral", "not_in_scope")


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
            "SEM_FILTER(mutual NDA, unilateral NDA, or confidentiality-and-standstill)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is categorized as a mutual non-disclosure "
                "agreement, a unilateral non-disclosure agreement, or a "
                "confidentiality-and-standstill agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(agreement category and operative direction)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "agreement_type": (
                        "exactly mutual_nda, unilateral_nda, or "
                        "confidentiality_standstill; use not_in_scope only for a "
                        "document outside those categories"
                    ),
                    "non_disclosure_direction": (
                        "exactly mutual or unilateral according to the operative "
                        "nondisclosure duty; use not_in_scope only for an out-of-scope "
                        "document"
                    ),
                },
            )
            extracted["agreement_type"] = extracted["agreement_type"].map(
                lambda value: normalize_enum(value, AGREEMENT_TYPES)
            )
            extracted["non_disclosure_direction"] = extracted[
                "non_disclosure_direction"
            ].map(lambda value: normalize_enum(value, DIRECTIONS))
            step.set_output(extracted)

        grouped = (
            extracted.groupby(
                ["agreement_type", "non_disclosure_direction"],
                sort=False,
                dropna=False,
            )
            .size()
            .rename("document_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([agreement_type, direction], COUNT(*))",
            len(extracted),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["document_count", "agreement_type", "non_disclosure_direction"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(document_count DESC, agreement_type ASC, direction ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
