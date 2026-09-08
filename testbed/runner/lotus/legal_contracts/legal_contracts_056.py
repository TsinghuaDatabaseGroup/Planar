#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-056."""

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

TASK_ID = "legal_contracts-056"
AGREEMENT_TYPES = (
    "mutual_nda",
    "unilateral_nda",
    "confidentiality_standstill",
    "not_in_scope",
)


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
            "SEM_CLASSIFY(target agreement-type labels)",
            input_rows=len(agreements),
        ) as step:
            labeled = agreements.sem_map(
                "Assign {text} to exactly one label based on the direction of "
                "confidentiality protection and whether it includes a transaction "
                "standstill. Return exactly one of mutual_nda, unilateral_nda, "
                "confidentiality_standstill, or not_in_scope.",
                suffix="agreement_type",
            )
            labeled["agreement_type"] = labeled["agreement_type"].map(
                lambda value: normalize_enum(value, AGREEMENT_TYPES)
            )
            step.set_output(labeled)

        projected = labeled[["agreement_type", "word_count"]].rename(
            columns={"word_count": "document_word_count"}
        )
        projected = projected.reset_index(drop=True)
        tracker.record(
            "PROJECT(agreement_type, WORD_COUNT(document_text) AS document_word_count)",
            len(labeled),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("agreement_type", sort=False, dropna=False)
            .agg(
                document_count=("agreement_type", "size"),
                avg_word_count=("document_word_count", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_word_count"] = grouped["avg_word_count"].round(2)
        tracker.record(
            "GROUP_BY([agreement_type], COUNT(*), ROUND(AVG(word_count), 2))",
            len(projected),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["document_count", "agreement_type"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(document_count DESC, agreement_type ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
