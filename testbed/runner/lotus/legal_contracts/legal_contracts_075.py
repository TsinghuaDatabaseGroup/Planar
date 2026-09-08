#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-075."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-075"


def _longest_non_solicitation(row):
    values = [
        row["employee_non_solicitation_months"],
        row["customer_non_solicitation_months"],
    ]
    return max(value for value in values if pd.notna(value))


def _comparison(row):
    if row["non_compete_months"] > row["longest_non_solicitation_months"]:
        return "non_compete_longer"
    if row["non_compete_months"] == row["longest_non_solicitation_months"]:
        return "equal"
    return "non_solicitation_longer"


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
            "SEM_FILTER(EX-10 material contract exhibit including dual-labelled)",
            input_rows=len(documents),
        ) as step:
            material_contracts = documents.sem_filter(
                "The document {text} is an EX-10 material contract exhibit, including "
                "a dual-labelled exhibit."
            ).reset_index(drop=True)
            step.set_output(material_contracts)

        with tracker.step(
            "SEM_FILTER(restrictive clauses, relief, and severability)",
            input_rows=len(material_contracts),
        ) as step:
            qualifying_clauses = material_contracts.sem_filter(
                "The agreement {text} expressly contains an operative non-compete "
                "clause, at least one operative employee or customer non-solicitation "
                "clause, injunctive-relief language, and severability language. All "
                "four conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying_clauses)

        with tracker.step(
            "SEM_EXTRACT(restrictive durations)",
            input_rows=len(qualifying_clauses),
        ) as step:
            extracted = qualifying_clauses.sem_extract(
                input_cols=["text"],
                output_cols={
                    "non_compete_months": (
                        "the stated non-compete duration normalized to months, or null "
                        "when unstated or unquantifiable"
                    ),
                    "employee_non_solicitation_months": (
                        "the stated employee-non-solicitation duration normalized to "
                        "months, or null when absent, unstated, or unquantifiable"
                    ),
                    "customer_non_solicitation_months": (
                        "the stated customer-non-solicitation duration normalized to "
                        "months, or null when absent, unstated, or unquantifiable"
                    ),
                },
            )
            for column in (
                "non_compete_months",
                "employee_non_solicitation_months",
                "customer_non_solicitation_months",
            ):
                extracted[column] = extracted[column].map(parse_number)
            step.set_output(extracted)

        complete = extracted.loc[
            extracted["non_compete_months"].notna()
            & (
                extracted["employee_non_solicitation_months"].notna()
                | extracted["customer_non_solicitation_months"].notna()
            )
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(non-compete duration and at least one non-solicitation duration present)",
            len(extracted),
            len(complete),
            output=complete,
        )

        projected = complete[["non_compete_months"]].copy()
        projected["longest_non_solicitation_months"] = [
            _longest_non_solicitation(row) for _, row in complete.iterrows()
        ]
        projected["comparison_class"] = [
            _comparison(row) for _, row in projected.iterrows()
        ]
        projected = projected.reset_index(drop=True)
        tracker.record(
            "PROJECT(normalized periods and comparison class)",
            len(complete),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("comparison_class", sort=False)
            .agg(
                agreement_count=("comparison_class", "size"),
                average_non_compete_months=("non_compete_months", "mean"),
                average_longest_non_solicitation_months=(
                    "longest_non_solicitation_months",
                    "mean",
                ),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["average_non_compete_months"] = grouped[
                "average_non_compete_months"
            ].round(2)
            grouped["average_longest_non_solicitation_months"] = grouped[
                "average_longest_non_solicitation_months"
            ].round(2)
        tracker.record(
            "GROUP_BY([comparison_class], count and average durations)",
            len(projected),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
