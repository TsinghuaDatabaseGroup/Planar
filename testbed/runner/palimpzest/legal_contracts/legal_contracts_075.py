#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-075."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_mixed_documents,
    memory_dataset,
    normalize_scalar_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-075"
DATASET = "contract-exhibit"
DURATION_COLUMNS = [
    "non_compete_months",
    "employee_non_solicitation_months",
    "customer_non_solicitation_months",
]


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def longest_non_solicitation(row: pd.Series) -> float:
    return max(
        value
        for value in (
            row["employee_non_solicitation_months"],
            row["customer_non_solicitation_months"],
        )
        if pd.notna(value)
    )


def comparison_class(row: pd.Series) -> str:
    if row["non_compete_months"] > row["longest_non_solicitation_months"]:
        return "non_compete_longer"
    if row["non_compete_months"] == row["longest_non_solicitation_months"]:
        return "equal"
    return "non_solicitation_longer"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is an EX-10 material contract exhibit, "
                "including a dual-labelled exhibit."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        clause_plan = memory_dataset(
            f"{TASK_ID}-clauses", scoped
        ).sem_filter(
            filter=(
                "Keep the agreement only if it expressly contains an operative "
                "non-compete clause, at least one operative employee or customer "
                "non-solicitation clause, injunctive-relief language, and severability "
                "language; all four conditions must hold."
            ),
            depends_on=["text"],
        )
        started = time.time()
        clause_result = clause_plan.run(config)
        qualifying_clauses = result_frame(clause_result, scoped)
        tracker.record_semantic(
            "sem_filter", len(scoped), qualifying_clauses, clause_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying_clauses
        ).sem_map(
            cols=[
                {"name": "non_compete_months", "type": float | None,
                 "desc": "The stated non-compete duration normalized to months, or null when unstated or not quantifiable."},
                {"name": "employee_non_solicitation_months", "type": float | None,
                 "desc": "The stated employee non-solicitation duration normalized to months, or null when absent, unstated, or not quantifiable."},
                {"name": "customer_non_solicitation_months", "type": float | None,
                 "desc": "The stated customer non-solicitation duration normalized to months, or null when absent, unstated, or not quantifiable."},
            ],
            desc="Extract and normalize the three restrictive periods to months.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result, qualifying_clauses, DURATION_COLUMNS
        )
        for column in DURATION_COLUMNS:
            extracted[column] = extracted[column].map(normalize_number)
        tracker.record_semantic(
            "sem_map", len(qualifying_clauses), extracted, extraction_result,
            time.time() - started,
        )

        complete = extracted.loc[
            extracted["non_compete_months"].notna()
            & (
                extracted["employee_non_solicitation_months"].notna()
                | extracted["customer_non_solicitation_months"].notna()
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), complete)

        projected = complete[["non_compete_months"]].copy()
        projected["longest_non_solicitation_months"] = complete.apply(
            longest_non_solicitation, axis=1
        )
        projected["comparison_class"] = projected.apply(comparison_class, axis=1)
        projected = projected[
            [
                "comparison_class",
                "non_compete_months",
                "longest_non_solicitation_months",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(complete), projected)

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
        for column in (
            "average_non_compete_months",
            "average_longest_non_solicitation_months",
        ):
            grouped[column] = grouped[column].round(2)
        tracker.record("groupby", len(projected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
