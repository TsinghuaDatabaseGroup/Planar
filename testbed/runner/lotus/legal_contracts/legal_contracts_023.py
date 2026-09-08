#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-023."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-023"
EXCEPTIONS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
)


def _greatest_ignore_null(row):
    values = [
        row["customer_non_solicitation_months"],
        row["employee_non_solicitation_months"],
    ]
    values = [value for value in values if pd.notna(value)]
    return max(values) if values else None


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
            "SEM_FILTER(standalone agreement with determinate standstill and non-solicitation)",
            input_rows=len(documents),
        ) as step:
            duration_candidates = documents.sem_filter(
                "The document {text} is itself a standalone non-disclosure or "
                "confidentiality agreement, contains a standstill provision with "
                "a determinate duration, and states a determinate duration for at "
                "least one customer- or employee-non-solicitation provision. "
                "Incidental references unrelated to the confidentiality duty do "
                "not count."
            ).reset_index(drop=True)
            step.set_output(duration_candidates)

        with tracker.step(
            "SEM_FILTER(exactly five of six confidentiality carve-outs)",
            input_rows=len(duration_candidates),
        ) as step:
            five_exception_agreements = duration_candidates.sem_filter(
                "The agreement {text} contains exactly five, not fewer or all six, "
                "of these operative confidentiality exceptions: public information, "
                "prior knowledge, independent development, lawful unrestricted "
                "third-party receipt, legally compelled disclosure, and disclosure "
                "authorized by the disclosing party."
            ).reset_index(drop=True)
            step.set_output(five_exception_agreements)

        with tracker.step(
            "SEM_EXTRACT(law, missing carve-out, and durations in months)",
            input_rows=len(five_exception_agreements),
        ) as step:
            extracted = five_exception_agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction"
                    ),
                    "missing_exception": (
                        "the single missing carve-out, exactly one of "
                        "public_information, prior_knowledge, independent_development, "
                        "third_party_receipt, legal_compulsion, or consent"
                    ),
                    "standstill_months": (
                        "the determinate standstill duration in months as an integer, "
                        "using 30 days per month and 12 months per year"
                    ),
                    "customer_non_solicitation_months": (
                        "the determinate customer-non-solicitation duration in months "
                        "as an integer using 30 days per month and 12 months per year, "
                        "or null if no determinate duration is stated"
                    ),
                    "employee_non_solicitation_months": (
                        "the determinate employee-non-solicitation duration in months "
                        "as an integer using 30 days per month and 12 months per year, "
                        "or null if no determinate duration is stated"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted["missing_exception"] = extracted["missing_exception"].map(
                lambda value: normalize_enum(value, EXCEPTIONS)
            )
            for column in (
                "standstill_months",
                "customer_non_solicitation_months",
                "employee_non_solicitation_months",
            ):
                extracted[column] = extracted[column].map(parse_number)
            step.set_output(extracted)

        projected = extracted.copy()
        projected["longest_solicitation_months"] = projected.apply(
            _greatest_ignore_null,
            axis=1,
        )
        projected = projected[
            [
                "document_id",
                "governing_law",
                "missing_exception",
                "standstill_months",
                "longest_solicitation_months",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(document_id, law, missing exception, durations)",
            len(extracted),
            len(projected),
            output=projected,
        )

        qualifying = projected.loc[
            projected["standstill_months"].notna()
            & projected["longest_solicitation_months"].notna()
            & (
                projected["standstill_months"]
                <= projected["longest_solicitation_months"]
            )
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(standstill_months <= longest_solicitation_months)",
            len(projected),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
