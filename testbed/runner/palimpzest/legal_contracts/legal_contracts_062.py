#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-062."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

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
    normalize_text_value,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-062"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "not_in_scope"}
    )


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def stable_mode_optional(values: pd.Series) -> str | None:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    highest = max(counts.values())
    return sorted(value for value, count in counts.items() if count == highest)[0]


def project_restriction(row) -> dict:
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
    return {
        "restriction_category": category,
        "effective_duration_years": duration,
        "governing_law": row["governing_law"],
    }


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is an employment agreement, "
                "standalone non-compete agreement, or standalone non-solicitation "
                "agreement containing an operative customer or employee "
                "non-solicitation clause."
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

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", agreements
        ).sem_map(
            cols=[
                {"name": "customers_present", "type": bool,
                 "desc": "True if an operative customer non-solicitation restriction is present."},
                {"name": "customers_duration_years", "type": float | None,
                 "desc": "The customer-non-solicitation duration in years, or null when absent, unstated, or unquantifiable."},
                {"name": "employees_present", "type": bool,
                 "desc": "True if an operative employee non-solicitation restriction is present."},
                {"name": "employees_duration_years", "type": float | None,
                 "desc": "The employee-non-solicitation duration in years, or null when absent, unstated, or unquantifiable."},
                {"name": "governing_law", "type": str | None,
                 "desc": "The expressly stated governing-law jurisdiction, or null when unstated."},
            ],
            desc=(
                "Extract customer and employee non-solicitation presence, their "
                "durations, and governing law."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "customers_present", "customers_duration_years", "employees_present",
            "employees_duration_years", "governing_law",
        ]
        extracted = result_frame(extraction_result, agreements, generated)
        for column in ("customers_present", "employees_present"):
            extracted[column] = extracted[column].map(parse_bool)
        for column in ("customers_duration_years", "employees_duration_years"):
            extracted[column] = extracted[column].map(normalize_number)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(agreements), extracted, extraction_result,
            time.time() - started,
        )

        projected = pd.DataFrame.from_records(
            [project_restriction(row) for _, row in extracted.iterrows()],
            columns=[
                "restriction_category", "effective_duration_years", "governing_law"
            ],
        )
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("restriction_category", sort=False)
            .agg(
                agreement_count=("restriction_category", "size"),
                avg_effective_duration_years=("effective_duration_years", "mean"),
                most_common_governing_law=("governing_law", stable_mode_optional),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_effective_duration_years"] = grouped[
                "avg_effective_duration_years"
            ].round(2)
        tracker.record("groupby", len(projected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
