#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-062."""

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

TASK_ID = "legal_contracts-062"


def _project_restriction(row):
    customers_present = row["customers_present"]
    employees_present = row["employees_present"]
    customer_duration = row["customers_duration_years"]
    employee_duration = row["employees_duration_years"]

    if customers_present and employees_present:
        category = "Both"
        duration = (
            max(customer_duration, employee_duration)
            if pd.notna(customer_duration) and pd.notna(employee_duration)
            else None
        )
    elif customers_present:
        category = "Customers Only"
        duration = customer_duration
    else:
        category = "Employees Only"
        duration = employee_duration
    return pd.Series(
        {
            "restriction_category": category,
            "effective_duration_years": duration,
            "governing_law": row["governing_law"],
        }
    )


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
            "SEM_FILTER(restrictive agreement with customer or employee non-solicitation)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is an employment agreement, standalone "
                "non-compete agreement, or standalone non-solicitation agreement "
                "containing an operative customer or employee non-solicitation clause."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(non-solicitation presence, durations, and governing law)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
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
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                },
            )
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
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        if extracted.empty:
            projected = pd.DataFrame(
                columns=[
                    "restriction_category",
                    "effective_duration_years",
                    "governing_law",
                ]
            )
        else:
            projected = extracted.apply(_project_restriction, axis=1).reset_index(
                drop=True
            )
        tracker.record(
            "PROJECT(restriction category, effective duration, governing law)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("restriction_category", sort=False)
            .agg(
                agreement_count=("restriction_category", "size"),
                avg_effective_duration_years=("effective_duration_years", "mean"),
                most_common_governing_law=(
                    "governing_law",
                    stable_mode_optional,
                ),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_effective_duration_years"] = grouped[
                "avg_effective_duration_years"
            ].round(2)
        tracker.record(
            "GROUP_BY([restriction_category], count, average duration, governing-law mode)",
            len(projected),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
