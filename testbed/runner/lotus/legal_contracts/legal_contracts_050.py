#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-050."""

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
    parse_label_list,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-050"
COVENANT_TYPES = (
    "customer_non_solicitation",
    "employee_non_solicitation",
    "standstill",
)


def _max_non_null(row):
    values = [
        row["customer_period_months"],
        row["employee_period_months"],
        row["standstill_period_months"],
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
            "SEM_FILTER(Delaware-governed agreement with customer non-solicitation)",
            input_rows=len(documents),
        ) as step:
            customer_restricted = documents.sem_filter(
                "The document {text} is an agreement expressly governed by Delaware "
                "law and contains an operative customer non-solicitation covenant."
            ).reset_index(drop=True)
            step.set_output(customer_restricted)

        with tracker.step(
            "SEM_FILTER(employee non-solicitation or standstill)",
            input_rows=len(customer_restricted),
        ) as step:
            qualifying = customer_restricted.sem_filter(
                "The agreement {text} also contains an operative employee "
                "non-solicitation covenant or an operative standstill provision."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(type, matched covenants, and covenant durations)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "agreement_type": "a concise stated agreement type",
                    "matched_covenant_types": (
                        "a list of the operative matched covenant types, using only "
                        "customer_non_solicitation, employee_non_solicitation, and "
                        "standstill"
                    ),
                    "customer_period_months": (
                        "the customer-non-solicitation duration in months, or null "
                        "when unstated or unquantifiable"
                    ),
                    "employee_period_months": (
                        "the employee-non-solicitation duration in months, or null "
                        "when absent, unstated, or unquantifiable"
                    ),
                    "standstill_period_months": (
                        "the standstill duration in months, or null when absent, "
                        "unstated, or unquantifiable"
                    ),
                },
            )
            extracted["agreement_type"] = extracted["agreement_type"].map(clean_text)
            extracted["matched_covenant_types"] = extracted[
                "matched_covenant_types"
            ].map(lambda value: parse_label_list(value, COVENANT_TYPES))
            for column in (
                "customer_period_months",
                "employee_period_months",
                "standstill_period_months",
            ):
                extracted[column] = extracted[column].map(parse_number)
            step.set_output(extracted)

        projected = extracted.copy()
        projected["longest_period_months"] = projected.apply(
            _max_non_null,
            axis=1,
        )
        projected = projected[
            [
                "document_id",
                "agreement_type",
                "matched_covenant_types",
                "longest_period_months",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(document_id, agreement_type, matched_covenant_types, MAX_NON_NULL durations)",
            len(extracted),
            len(projected),
            output=projected,
        )

        qualifying_periods = projected.loc[
            projected["longest_period_months"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(longest_period_months IS NOT NULL)",
            len(projected),
            len(qualifying_periods),
            output=qualifying_periods,
        )

        answer = df_records(qualifying_periods)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
