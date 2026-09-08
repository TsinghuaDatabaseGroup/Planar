#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-066."""

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
    normalize_enum,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-066"
DATASET = "contract-exhibit"
EXCEPTION_COLUMNS = (
    "public_information", "prior_knowledge", "independent_development",
    "lawful_third_party_receipt", "legal_compulsion", "consent",
)
TERM_TYPES = ("finite", "perpetual", "unspecified")


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def exception_bracket(count) -> str:
    if count <= 1:
        return "0-1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    return "6"


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

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", agreements
        ).sem_map(
            cols=[
                *[
                    {"name": column, "type": bool,
                     "desc": f"True if the operative {column.replace('_', ' ')} confidentiality carve-out is present; false otherwise."}
                    for column in EXCEPTION_COLUMNS
                ],
                {"name": "term_type", "type": str | None,
                 "desc": "Exactly finite, perpetual, or unspecified for the confidentiality term."},
                {"name": "finite_duration_years", "type": float | None,
                 "desc": "The finite confidentiality duration in years, or null for a perpetual, unspecified, or unquantifiable term."},
                {"name": "return_of_materials", "type": bool,
                 "desc": "True if return or destruction of confidential materials is required; false otherwise."},
            ],
            desc=(
                "Extract six confidentiality-carve-out indicators, term type and "
                "finite duration, and the return-of-materials indicator."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [*EXCEPTION_COLUMNS, "term_type", "finite_duration_years", "return_of_materials"]
        extracted = result_frame(extraction_result, agreements, generated)
        for column in EXCEPTION_COLUMNS:
            extracted[column] = extracted[column].map(parse_bool)
        extracted["term_type"] = extracted["term_type"].map(
            lambda value: normalize_enum(value, TERM_TYPES)
        )
        extracted["finite_duration_years"] = extracted[
            "finite_duration_years"
        ].map(normalize_number)
        extracted["return_of_materials"] = extracted["return_of_materials"].map(
            parse_bool
        )
        tracker.record_semantic(
            "sem_map", len(agreements), extracted, extraction_result,
            time.time() - started,
        )

        exception_counts = extracted[list(EXCEPTION_COLUMNS)].sum(axis=1)
        projected = extracted[["return_of_materials"]].copy()
        projected["exception_bracket"] = exception_counts.map(exception_bracket)
        projected["finite_duration_years"] = extracted[
            "finite_duration_years"
        ].where(extracted["term_type"] == "finite")
        projected = projected[
            ["exception_bracket", "finite_duration_years", "return_of_materials"]
        ].reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("exception_bracket", sort=False)
            .agg(
                agreement_count=("exception_bracket", "size"),
                avg_finite_duration_years=("finite_duration_years", "mean"),
                return_of_materials_rate=("return_of_materials", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_finite_duration_years"] = grouped[
                "avg_finite_duration_years"
            ].round(2)
            grouped["return_of_materials_rate"] = grouped[
                "return_of_materials_rate"
            ].round(4)
        tracker.record("groupby", len(projected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
