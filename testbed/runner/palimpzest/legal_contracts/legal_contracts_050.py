#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-050."""

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
    normalize_text_value,
    parse_label_list,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-050"
DATASET = "contract-exhibit"
COVENANT_TYPES = (
    "customer_non_solicitation",
    "employee_non_solicitation",
    "standstill",
)


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        customer_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is an agreement expressly "
                "governed by Delaware law and contains an operative customer "
                "non-solicitation covenant."
            ),
            depends_on=["text"],
        )
        started = time.time()
        customer_result = customer_plan.run(config)
        customer_restricted = result_frame(customer_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            customer_restricted,
            customer_result,
            time.time() - started,
        )

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", customer_restricted
        ).sem_filter(
            filter=(
                "Keep the agreement only if it also contains an operative "
                "employee non-solicitation covenant or an operative standstill "
                "provision."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, customer_restricted)
        tracker.record_semantic(
            "sem_filter",
            len(customer_restricted),
            qualifying,
            condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {
                    "name": "agreement_type",
                    "type": str | None,
                    "desc": "A concise stated agreement type.",
                },
                {
                    "name": "matched_covenant_types",
                    "type": list[str],
                    "desc": (
                        "The operative matched covenant types, using only "
                        "customer_non_solicitation, employee_non_solicitation, "
                        "and standstill."
                    ),
                },
                {
                    "name": "customer_period_months",
                    "type": float | None,
                    "desc": (
                        "The customer-non-solicitation duration in months, or "
                        "null when unstated or unquantifiable."
                    ),
                },
                {
                    "name": "employee_period_months",
                    "type": float | None,
                    "desc": (
                        "The employee-non-solicitation duration in months, or "
                        "null when absent, unstated, or unquantifiable."
                    ),
                },
                {
                    "name": "standstill_period_months",
                    "type": float | None,
                    "desc": (
                        "The standstill duration in months, or null when absent, "
                        "unstated, or unquantifiable."
                    ),
                },
            ],
            desc=(
                "Extract agreement type, matched covenant types, and each "
                "matched covenant duration normalized to months."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "agreement_type",
            "matched_covenant_types",
            "customer_period_months",
            "employee_period_months",
            "standstill_period_months",
        ]
        extracted = result_frame(extraction_result, qualifying, generated)
        extracted["agreement_type"] = extracted["agreement_type"].map(
            normalize_text
        )
        extracted["matched_covenant_types"] = extracted[
            "matched_covenant_types"
        ].map(lambda value: parse_label_list(value, COVENANT_TYPES))
        duration_columns = [
            "customer_period_months",
            "employee_period_months",
            "standstill_period_months",
        ]
        for column in duration_columns:
            extracted[column] = extracted[column].map(normalize_number)
        tracker.record_semantic(
            "sem_map",
            len(qualifying),
            extracted,
            extraction_result,
            time.time() - started,
        )

        projected = extracted.copy()
        projected["longest_period_months"] = projected[
            duration_columns
        ].max(axis=1, skipna=True)
        projected = projected[
            [
                "document_id",
                "agreement_type",
                "matched_covenant_types",
                "longest_period_months",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        qualifying_periods = projected.loc[
            projected["longest_period_months"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(projected), qualifying_periods)

        answer = df_records(qualifying_periods)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
