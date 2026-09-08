#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-066."""

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
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-066"
EXCEPTION_COLUMNS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "lawful_third_party_receipt",
    "legal_compulsion",
    "consent",
)
TERM_TYPES = ("finite", "perpetual", "unspecified")


def _exception_bracket(count):
    if count <= 1:
        return "0-1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    return "6"


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
            "SEM_EXTRACT(exceptions, term, and return-of-materials requirement)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
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
                    "term_type": (
                        "exactly finite, perpetual, or unspecified for the "
                        "confidentiality term"
                    ),
                    "finite_duration_years": (
                        "the finite confidentiality duration normalized to years, or "
                        "null for a perpetual, unspecified, or unquantifiable term"
                    ),
                    "return_of_materials": (
                        "true if return or destruction of confidential materials is "
                        "required; false otherwise"
                    ),
                },
            )
            for column in EXCEPTION_COLUMNS:
                extracted[column] = extracted[column].map(parse_bool)
            extracted["term_type"] = extracted["term_type"].map(
                lambda value: normalize_enum(value, TERM_TYPES)
            )
            extracted["finite_duration_years"] = extracted[
                "finite_duration_years"
            ].map(parse_number)
            extracted["return_of_materials"] = extracted["return_of_materials"].map(
                parse_bool
            )
            step.set_output(extracted)

        exception_counts = extracted[list(EXCEPTION_COLUMNS)].sum(axis=1)
        projected = extracted[["return_of_materials"]].copy()
        projected["exception_bracket"] = exception_counts.map(_exception_bracket)
        projected["finite_duration_years"] = extracted[
            "finite_duration_years"
        ].where(extracted["term_type"] == "finite")
        projected = projected[
            ["exception_bracket", "finite_duration_years", "return_of_materials"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(exception bracket, finite-only duration, return requirement)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("exception_bracket", sort=False)
            .agg(
                agreement_count=("exception_bracket", "size"),
                avg_finite_duration_years=("finite_duration_years", "mean"),
                return_of_materials_rate=("return_of_materials", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_finite_duration_years"] = grouped[
                "avg_finite_duration_years"
            ].round(2)
            grouped["return_of_materials_rate"] = grouped[
                "return_of_materials_rate"
            ].round(4)
        tracker.record(
            "GROUP_BY([exception_bracket], count, finite-term average, return rate)",
            len(projected),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
