#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-037."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    parse_bool,
    parse_number,
    save_output,
    setup,
    stable_mode_optional,
)

TASK_ID = "legal_contracts-037"
TERM_TYPE_ORDER = ("fixed_short", "fixed_long", "perpetual", "unspecified")


def _classify_term(row):
    if row["is_perpetual"]:
        return "perpetual"
    duration = row["term_duration_months"]
    if pd.notna(duration) and duration <= 36:
        return "fixed_short"
    if pd.notna(duration) and duration > 36:
        return "fixed_long"
    return "unspecified"


def main():
    setup(max_tokens=768, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_EXTRACT(confidentiality term, perpetual status, and governing law)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "term_duration_months": (
                        "the single quantifiable fixed confidentiality duration in "
                        "months, or null when no one fixed duration is quantifiable"
                    ),
                    "is_perpetual": (
                        "true if the confidentiality duty is perpetual or indefinite, "
                        "otherwise false"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                },
            )
            extracted["term_duration_months"] = extracted[
                "term_duration_months"
            ].map(parse_number)
            extracted["is_perpetual"] = extracted["is_perpetual"].map(parse_bool)
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        projected = extracted[["governing_law"]].copy()
        projected["term_type"] = extracted.apply(_classify_term, axis=1)
        projected = projected[["term_type", "governing_law"]].reset_index(drop=True)
        tracker.record(
            "PROJECT(CASE confidentiality term AS term_type, governing_law)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("term_type", sort=False, dropna=False)
            .agg(
                agreement_count=("term_type", "size"),
                most_common_governing_law=(
                    "governing_law",
                    stable_mode_optional,
                ),
            )
            .reset_index()
        )
        order = {term_type: index for index, term_type in enumerate(TERM_TYPE_ORDER)}
        grouped = grouped.sort_values(
            "term_type",
            key=lambda values: values.map(order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([term_type], COUNT(*), MODE(governing_law))",
            len(projected),
            len(grouped),
            output=grouped,
        )

        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
