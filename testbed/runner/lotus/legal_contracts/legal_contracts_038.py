#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-038."""

import os
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
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-038"


def _greatest_non_null(row):
    values = [
        row["non_compete_months"],
        row["customer_non_solicitation_months"],
        row["employee_non_solicitation_months"],
    ]
    values = [value for value in values if pd.notna(value)]
    return max(values) if values else None


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
            "SEM_FILTER(NDA or EX-10 material contract exhibit)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a non-disclosure agreement or an EX-10 "
                "material contract exhibit."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(family, law, term, and restriction periods)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "exhibit_family": (
                        "the exhibit family; use NDA for an NDA, the applicable EX-xx "
                        "label for a material contract exhibit, and NDA|EX-xx when the "
                        "document has both statuses"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                    "term_months": (
                        "the finite confidentiality term normalized to months, or "
                        "null when unstated, perpetual, or unquantifiable"
                    ),
                    "non_compete_months": (
                        "the quantifiable non-compete period in months, or null when "
                        "absent or unquantifiable"
                    ),
                    "customer_non_solicitation_months": (
                        "the quantifiable customer-non-solicitation period in months, "
                        "or null when absent or unquantifiable"
                    ),
                    "employee_non_solicitation_months": (
                        "the quantifiable employee-non-solicitation period in months, "
                        "or null when absent or unquantifiable"
                    ),
                },
            )
            extracted["exhibit_family"] = extracted["exhibit_family"].map(clean_text)
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            for column in (
                "term_months",
                "non_compete_months",
                "customer_non_solicitation_months",
                "employee_non_solicitation_months",
            ):
                extracted[column] = extracted[column].map(parse_number)
            step.set_output(extracted)

        projected = extracted.copy()
        projected["longest_restriction_months"] = projected.apply(
            _greatest_non_null,
            axis=1,
        )
        projected = projected[
            [
                "document_id",
                "exhibit_family",
                "governing_law",
                "term_months",
                "longest_restriction_months",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(document_id, family, law, term, GREATEST_NON_NULL restrictions)",
            len(extracted),
            len(projected),
            output=projected,
        )

        qualifying = projected.loc[
            projected["governing_law"].notna()
            & projected["term_months"].notna()
            & projected["longest_restriction_months"].notna()
            & (
                projected["term_months"]
                == projected["longest_restriction_months"]
            )
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(governing_law IS NOT NULL AND term_months IS NOT NULL AND term_months = longest_restriction_months)",
            len(projected),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
