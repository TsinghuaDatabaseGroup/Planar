#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-037."""

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

TASK_ID = "legal_contracts-037"
DATASET = "contract-exhibit"
TERM_TYPE_ORDER = ("fixed_short", "fixed_long", "perpetual", "unspecified")


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def classify_term(row) -> str:
    if row["is_perpetual"]:
        return "perpetual"
    duration = row["term_duration_months"]
    if pd.notna(duration) and duration <= 36:
        return "fixed_short"
    if pd.notna(duration) and duration > 36:
        return "fixed_long"
    return "unspecified"


def stable_mode_optional(values: pd.Series) -> str | None:
    cleaned = [
        str(value)
        for value in values
        if pd.notna(value) and str(value).strip()
    ]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    highest = max(counts.values())
    return sorted(
        value for value, count in counts.items() if count == highest
    )[0]


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
            "sem_filter",
            len(documents),
            agreements,
            agreement_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", agreements
        ).sem_map(
            cols=[
                {
                    "name": "term_duration_months",
                    "type": float | None,
                    "desc": (
                        "The single quantifiable fixed confidentiality duration "
                        "in months, or null when no one fixed duration is "
                        "quantifiable."
                    ),
                },
                {
                    "name": "is_perpetual",
                    "type": bool,
                    "desc": (
                        "True if the confidentiality duty is perpetual or "
                        "indefinite; false otherwise."
                    ),
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The expressly stated governing-law jurisdiction, or "
                        "null when unstated."
                    ),
                },
            ],
            desc=(
                "Extract the fixed confidentiality duration, perpetual status, "
                "and governing law."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            agreements,
            ["term_duration_months", "is_perpetual", "governing_law"],
        )
        extracted["term_duration_months"] = extracted[
            "term_duration_months"
        ].map(normalize_number)
        extracted["is_perpetual"] = extracted["is_perpetual"].map(parse_bool)
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        tracker.record_semantic(
            "sem_map",
            len(agreements),
            extracted,
            extraction_result,
            time.time() - started,
        )

        projected = extracted[["governing_law"]].copy()
        projected["term_type"] = pd.Series(
            (classify_term(row) for _, row in extracted.iterrows()),
            index=extracted.index,
            dtype="object",
        )
        projected = projected[
            ["term_type", "governing_law"]
        ].reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("term_type", sort=False, dropna=False)
            .agg(
                agreement_count=("term_type", "size"),
                most_common_governing_law=(
                    "governing_law",
                    stable_mode_optional,
                ),
            )
            .reset_index()
        )
        order = {
            term_type: index
            for index, term_type in enumerate(TERM_TYPE_ORDER)
        }
        grouped = grouped.sort_values(
            "term_type",
            key=lambda values: values.map(order),
        ).reset_index(drop=True)
        tracker.record("groupby", len(projected), grouped)

        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
