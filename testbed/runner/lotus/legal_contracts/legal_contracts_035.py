#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-035."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-035"


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
            "SEM_FILTER(non-disclosure agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a non-disclosure agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(quantifiable non-compete and employee non-solicitation with law)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} contains operative non-compete and employee "
                "non-solicitation clauses, both with quantifiable fixed periods, and "
                "states a governing-law provision. All conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(document name, law, and restriction periods)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "document_name": "the stated name or title of the agreement",
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction or rule"
                    ),
                    "non_compete_months": (
                        "the fixed non-compete period normalized to months as a number"
                    ),
                    "employee_non_solicitation_months": (
                        "the fixed employee-non-solicitation period normalized to "
                        "months as a number"
                    ),
                },
            )
            extracted["document_name"] = extracted["document_name"].map(clean_text)
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted["non_compete_months"] = extracted[
                "non_compete_months"
            ].map(parse_number)
            extracted["employee_non_solicitation_months"] = extracted[
                "employee_non_solicitation_months"
            ].map(parse_number)
            step.set_output(extracted)

        projected = extracted[
            [
                "document_name",
                "governing_law",
                "non_compete_months",
                "employee_non_solicitation_months",
            ]
        ].copy()
        projected["difference_months"] = (
            projected["non_compete_months"]
            - projected["employee_non_solicitation_months"]
        )
        projected = projected[
            ["document_name", "governing_law", "difference_months"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(document_name, governing_law, non_compete_months - employee_non_solicitation_months AS difference_months)",
            len(extracted),
            len(projected),
            output=projected,
        )

        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
