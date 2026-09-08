#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-065."""

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

TASK_ID = "legal_contracts-065"
DATASET = "contract-exhibit"
EXCEPTION_COLUMNS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "lawful_third_party_receipt",
    "legal_compulsion",
    "consent",
)


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "not_in_scope"}
    )


def normalize_iso_date(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def execution_decade(value) -> str | None:
    if value is None:
        return None
    year = int(str(value)[:4])
    return f"{(year // 10) * 10}s"


def stable_mode_optional(values: pd.Series) -> str | None:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    highest = max(counts.values())
    return sorted(value for value, count in counts.items() if count == highest)[0]


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
                "confidentiality-and-standstill agreement with an identifiable "
                "effective date from 1990 through 2029, inclusive."
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
                {"name": "effective_date", "type": str | None,
                 "desc": "The explicitly stated effective date as YYYY-MM-DD."},
                {"name": "governing_law", "type": str | None,
                 "desc": "The expressly stated governing-law jurisdiction, or null when unstated."},
                {"name": "is_perpetual", "type": bool,
                 "desc": "True if the confidentiality term is expressly perpetual or indefinite; false otherwise."},
                *[
                    {"name": column, "type": bool,
                     "desc": f"True if the operative {column.replace('_', ' ')} confidentiality carve-out is present; false otherwise."}
                    for column in EXCEPTION_COLUMNS
                ],
            ],
            desc=(
                "Extract the effective date, governing law, perpetual-term status, "
                "and presence of the six standard confidentiality carve-outs."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["effective_date", "governing_law", "is_perpetual", *EXCEPTION_COLUMNS]
        extracted = result_frame(extraction_result, agreements, generated)
        extracted["effective_date"] = extracted["effective_date"].map(normalize_iso_date)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        extracted["is_perpetual"] = extracted["is_perpetual"].map(parse_bool)
        for column in EXCEPTION_COLUMNS:
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(agreements), extracted, extraction_result,
            time.time() - started,
        )

        projected = extracted[["governing_law", "is_perpetual"]].copy()
        projected["execution_decade"] = extracted["effective_date"].map(
            execution_decade
        )
        projected["exception_count"] = extracted[list(EXCEPTION_COLUMNS)].sum(axis=1)
        projected = projected[
            ["execution_decade", "exception_count", "governing_law", "is_perpetual"]
        ].reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("execution_decade", sort=False, dropna=False)
            .agg(
                agreement_count=("execution_decade", "size"),
                avg_exception_count=("exception_count", "mean"),
                most_common_governing_law=("governing_law", stable_mode_optional),
                perpetual_term_rate=("is_perpetual", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_exception_count"] = grouped["avg_exception_count"].round(2)
            grouped["perpetual_term_rate"] = grouped["perpetual_term_rate"].round(4)
        tracker.record("groupby", len(projected), grouped)

        ordered = grouped.sort_values(
            "execution_decade", kind="stable"
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
