#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-063."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-063"
DATASET = "contract-exhibit"


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def duration_pattern(row) -> str:
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
    longest = max(non_solicitation_durations)
    if noncompete > longest:
        return "NC Longer"
    if noncompete < longest:
        return "NS Longer"
    return "Equal"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure "
                "agreement, unilateral non-disclosure agreement, or "
                "confidentiality-and-standstill agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        agreement_result = agreement_plan.run(config)
        agreements = result_frame(agreement_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), agreements, agreement_result,
            time.time() - started,
        )

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains an operative non-compete "
                "clause and at least one operative customer or employee "
                "non-solicitation clause."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, agreements)
        tracker.record_semantic(
            "sem_filter", len(agreements), qualifying, condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {"name": "noncompete_duration_years", "type": float | None,
                 "desc": "The non-compete duration in years, or null when unstated or unquantifiable."},
                {"name": "customers_present", "type": bool,
                 "desc": "True if an operative customer non-solicitation restriction is present."},
                {"name": "customers_duration_years", "type": float | None,
                 "desc": "The customer-non-solicitation duration in years, or null when absent, unstated, or unquantifiable."},
                {"name": "employees_present", "type": bool,
                 "desc": "True if an operative employee non-solicitation restriction is present."},
                {"name": "employees_duration_years", "type": float | None,
                 "desc": "The employee-non-solicitation duration in years, or null when absent, unstated, or unquantifiable."},
            ],
            desc="Extract non-compete and non-solicitation presence and durations.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "noncompete_duration_years", "customers_present",
            "customers_duration_years", "employees_present",
            "employees_duration_years",
        ]
        extracted = result_frame(extraction_result, qualifying, generated)
        for column in (
            "noncompete_duration_years", "customers_duration_years",
            "employees_duration_years",
        ):
            extracted[column] = extracted[column].map(normalize_number)
        for column in ("customers_present", "employees_present"):
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(qualifying), extracted, extraction_result,
            time.time() - started,
        )

        projected = pd.DataFrame(
            {"duration_pattern": [
                duration_pattern(row) for _, row in extracted.iterrows()
            ]}
        )
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("duration_pattern", sort=False)
            .size()
            .rename("document_count")
            .reset_index()
        )
        tracker.record("groupby", len(projected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
