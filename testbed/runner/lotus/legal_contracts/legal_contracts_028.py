#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-028."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-028"


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
            "SEM_FILTER(standstill, employee non-solicitation, and no non-compete)",
            input_rows=len(agreements),
        ) as step:
            qualifying_clauses = agreements.sem_filter(
                "The agreement {text} contains operative standstill and employee "
                "non-solicitation restrictions and contains no operative non-compete "
                "restriction. All conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying_clauses)

        with tracker.step(
            "SEM_EXTRACT(law and normalized standstill/non-solicitation periods)",
            input_rows=len(qualifying_clauses),
        ) as step:
            extracted = qualifying_clauses.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                    "standstill_months": (
                        "the standstill period in months; multiply years by 12 and "
                        "convert a fixed-date interval as elapsed_days * 12 / 365, "
                        "rounded to 2 decimals; return null if unquantifiable"
                    ),
                    "employee_non_solicitation_months": (
                        "the employee-non-solicitation period in months; multiply "
                        "years by 12 and convert a fixed-date interval as elapsed_days "
                        "* 12 / 365, rounded to 2 decimals; return null if unquantifiable"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            extracted["standstill_months"] = extracted["standstill_months"].map(
                parse_number
            )
            extracted["employee_non_solicitation_months"] = extracted[
                "employee_non_solicitation_months"
            ].map(parse_number)
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["standstill_months"].notna()
            & extracted["employee_non_solicitation_months"].notna()
            & (
                extracted["standstill_months"]
                <= extracted["employee_non_solicitation_months"]
            ),
            [
                "document_id",
                "governing_law",
                "standstill_months",
                "employee_non_solicitation_months",
            ],
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(standstill_months IS NOT NULL AND employee_non_solicitation_months IS NOT NULL AND standstill_months <= employee_non_solicitation_months)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
