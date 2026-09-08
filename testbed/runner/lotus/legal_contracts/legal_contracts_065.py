#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-065."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    normalize_iso_date,
    parse_bool,
    save_output,
    setup,
    stable_mode_optional,
)

TASK_ID = "legal_contracts-065"
EXCEPTION_COLUMNS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "lawful_third_party_receipt",
    "legal_compulsion",
    "consent",
)


def _execution_decade(value):
    if value is None:
        return None
    year = int(str(value)[:4])
    return f"{(year // 10) * 10}s"


def main():
    setup(max_tokens=1024, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(dated NDA from 1990 through 2029)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement with an identifiable effective date from "
                "1990 through 2029, inclusive."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(date, law, perpetual term, and six exceptions)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "effective_date": (
                        "the explicitly stated effective date formatted as YYYY-MM-DD"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                    "is_perpetual": (
                        "true if the confidentiality term is expressly perpetual or "
                        "indefinite; false otherwise"
                    ),
                    "public_information": (
                        "true if an operative public-information exception is present"
                    ),
                    "prior_knowledge": (
                        "true if an operative prior-knowledge exception is present"
                    ),
                    "independent_development": (
                        "true if an operative independent-development exception is present"
                    ),
                    "lawful_third_party_receipt": (
                        "true if an operative lawful unrestricted third-party-receipt "
                        "exception is present"
                    ),
                    "legal_compulsion": (
                        "true if an operative legal-compulsion exception is present"
                    ),
                    "consent": (
                        "true if an operative disclosure-with-consent exception is present"
                    ),
                },
            )
            extracted["effective_date"] = extracted["effective_date"].map(
                normalize_iso_date
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            extracted["is_perpetual"] = extracted["is_perpetual"].map(parse_bool)
            for column in EXCEPTION_COLUMNS:
                extracted[column] = extracted[column].map(parse_bool)
            step.set_output(extracted)

        projected = extracted[["governing_law", "is_perpetual"]].copy()
        projected["execution_decade"] = extracted["effective_date"].map(
            _execution_decade
        )
        projected["exception_count"] = extracted[list(EXCEPTION_COLUMNS)].sum(axis=1)
        projected = projected[
            [
                "execution_decade",
                "exception_count",
                "governing_law",
                "is_perpetual",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(decade, exception count, law, perpetual status)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("execution_decade", sort=False, dropna=False)
            .agg(
                agreement_count=("execution_decade", "size"),
                avg_exception_count=("exception_count", "mean"),
                most_common_governing_law=(
                    "governing_law",
                    stable_mode_optional,
                ),
                perpetual_term_rate=("is_perpetual", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_exception_count"] = grouped[
                "avg_exception_count"
            ].round(2)
            grouped["perpetual_term_rate"] = grouped[
                "perpetual_term_rate"
            ].round(4)
        tracker.record(
            "GROUP_BY([execution_decade], count, exception average, law mode, perpetual rate)",
            len(projected),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values("execution_decade", kind="stable").reset_index(
            drop=True
        )
        tracker.record(
            "ORDER_BY(execution_decade ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
