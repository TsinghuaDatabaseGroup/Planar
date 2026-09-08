#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-063."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_bool,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-063"


def _duration_pattern(row):
    noncompete = row["noncompete_duration_years"]
    customer_duration = row["customers_duration_years"]
    employee_duration = row["employees_duration_years"]
    if (
        pd.isna(noncompete)
        or (row["customers_present"] and pd.isna(customer_duration))
        or (row["employees_present"] and pd.isna(employee_duration))
    ):
        return "Partially Unspecified"

    non_solicitation_durations = [
        duration
        for duration in (customer_duration, employee_duration)
        if pd.notna(duration)
    ]
    if not non_solicitation_durations:
        return "Partially Unspecified"
    longest_non_solicitation = max(non_solicitation_durations)
    if noncompete > longest_non_solicitation:
        return "NC Longer"
    if noncompete < longest_non_solicitation:
        return "NS Longer"
    return "Equal"


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
            "SEM_FILTER(non-compete and customer or employee non-solicitation)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} contains an operative non-compete clause and "
                "at least one operative customer or employee non-solicitation clause."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(non-compete and non-solicitation durations)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "noncompete_duration_years": (
                        "the non-compete duration normalized to years, or null when "
                        "unstated or unquantifiable"
                    ),
                    "customers_present": (
                        "true if an operative customer non-solicitation restriction "
                        "is present"
                    ),
                    "customers_duration_years": (
                        "the customer-non-solicitation duration normalized to years, "
                        "or null when absent, unstated, or unquantifiable"
                    ),
                    "employees_present": (
                        "true if an operative employee non-solicitation restriction "
                        "is present"
                    ),
                    "employees_duration_years": (
                        "the employee-non-solicitation duration normalized to years, "
                        "or null when absent, unstated, or unquantifiable"
                    ),
                },
            )
            extracted["noncompete_duration_years"] = extracted[
                "noncompete_duration_years"
            ].map(parse_number)
            extracted["customers_present"] = extracted["customers_present"].map(
                parse_bool
            )
            extracted["employees_present"] = extracted["employees_present"].map(
                parse_bool
            )
            extracted["customers_duration_years"] = extracted[
                "customers_duration_years"
            ].map(parse_number)
            extracted["employees_duration_years"] = extracted[
                "employees_duration_years"
            ].map(parse_number)
            step.set_output(extracted)

        projected = extracted.apply(_duration_pattern, axis=1).to_frame(
            "duration_pattern"
        )
        projected = projected.reset_index(drop=True)
        tracker.record(
            "PROJECT(relative duration pattern)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("duration_pattern", sort=False)
            .size()
            .rename("document_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([duration_pattern], COUNT(*))",
            len(projected),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
