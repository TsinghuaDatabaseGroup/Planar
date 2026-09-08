#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-069."""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    clean_text,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-069"
TERM_TYPES = ("finite", "perpetual", "unspecified")


def _normalize_title(value):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def _sorted_distinct_numbers(values):
    return sorted({float(value) for value in values if pd.notna(value)})


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
            "SEM_FILTER(in-scope agreement with identifiable title)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement with an identifiable agreement title."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(title, confidentiality term, and governing law)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "agreement_title": "the identifiable stated agreement title",
                    "term_type": (
                        "exactly finite, perpetual, or unspecified for the "
                        "confidentiality term"
                    ),
                    "finite_term_years": (
                        "the finite confidentiality duration normalized to years, or "
                        "null when the term is perpetual, unspecified, or unquantifiable"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law value, or null when unstated"
                    ),
                },
            )
            extracted["agreement_title"] = extracted["agreement_title"].map(clean_text)
            extracted["term_type"] = extracted["term_type"].map(
                lambda value: normalize_enum(value, TERM_TYPES)
            )
            extracted["finite_term_years"] = extracted[
                "finite_term_years"
            ].map(parse_number)
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        projected = extracted[["governing_law"]].copy()
        projected["normalized_agreement_title"] = extracted["agreement_title"].map(
            _normalize_title
        )
        projected["finite_term_years"] = extracted["finite_term_years"].where(
            extracted["term_type"] == "finite"
        )
        if not projected.empty:
            projected["finite_term_years"] = projected[
                "finite_term_years"
            ].round(4)
        projected["is_perpetual"] = (extracted["term_type"] == "perpetual").astype(
            int
        )
        projected = projected[
            [
                "normalized_agreement_title",
                "finite_term_years",
                "is_perpetual",
                "governing_law",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(normalized title, rounded finite term, perpetual flag, law)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("normalized_agreement_title", sort=False)
            .agg(
                family_size=("normalized_agreement_title", "size"),
                distinct_finite_term_years=(
                    "finite_term_years",
                    _sorted_distinct_numbers,
                ),
                perpetual_member_count=("is_perpetual", "sum"),
                distinct_stated_governing_law_count=("governing_law", "nunique"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([normalized title], term and governing-law diversity)",
            len(projected),
            len(grouped),
            output=grouped,
        )

        large_families = grouped.loc[grouped["family_size"] >= 5].reset_index(
            drop=True
        )
        tracker.record(
            "FILTER(family_size >= 5)",
            len(grouped),
            len(large_families),
            output=large_families,
        )

        ordered = large_families.sort_values(
            ["family_size", "normalized_agreement_title"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(family_size DESC, normalized title ASC)",
            len(large_families),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
