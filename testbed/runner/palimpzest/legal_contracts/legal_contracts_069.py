#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-069."""

from __future__ import annotations

import os
import re
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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-069"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_title(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()) or None


def normalize_term_type(value) -> str:
    normalized = (normalize_text(value) or "unspecified").casefold()
    if normalized in {"finite", "perpetual", "unspecified"}:
        return normalized
    return "unspecified"


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
        return float(match.group()) if match else None


def sorted_distinct_numbers(values) -> list[float]:
    return sorted({float(value) for value in values if pd.notna(value)})


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure agreement, "
                "a unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement, and it has an identifiable agreement title."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", scoped
        ).sem_map(
            cols=[
                {
                    "name": "agreement_title",
                    "type": str,
                    "desc": "The identifiable agreement title stated in the document.",
                },
                {
                    "name": "term_type",
                    "type": str,
                    "desc": (
                        "Exactly finite, perpetual, or unspecified for the "
                        "confidentiality obligation."
                    ),
                },
                {
                    "name": "finite_term_years",
                    "type": float | None,
                    "desc": (
                        "The finite confidentiality duration normalized to years, or "
                        "null when the term is perpetual, unspecified, or not quantifiable."
                    ),
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The expressly stated governing-law jurisdiction, or null when "
                        "no governing law is stated."
                    ),
                },
            ],
            desc=(
                "Extract the agreement title, confidentiality-term type and finite "
                "duration, and expressly stated governing law."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            scoped,
            ["agreement_title", "term_type", "finite_term_years", "governing_law"],
        )
        extracted["agreement_title"] = extracted["agreement_title"].map(normalize_text)
        extracted["term_type"] = extracted["term_type"].map(normalize_term_type)
        extracted["finite_term_years"] = extracted["finite_term_years"].map(
            normalize_number
        )
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(scoped), extracted, extraction_result, time.time() - started
        )

        projected = pd.DataFrame(
            {
                "normalized_agreement_title": extracted["agreement_title"].map(
                    normalize_title
                ),
                "finite_term_years": extracted["finite_term_years"].where(
                    extracted["term_type"].eq("finite")
                ).round(4),
                "is_perpetual": extracted["term_type"].eq("perpetual").astype(int),
                "governing_law": extracted["governing_law"],
            }
        ).reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("normalized_agreement_title", sort=False, dropna=False)
            .agg(
                family_size=("normalized_agreement_title", "size"),
                distinct_finite_term_years=(
                    "finite_term_years",
                    sorted_distinct_numbers,
                ),
                perpetual_member_count=("is_perpetual", "sum"),
                distinct_stated_governing_law_count=("governing_law", "nunique"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(projected), grouped)

        frequent = grouped.loc[grouped["family_size"] >= 5].reset_index(drop=True)
        tracker.record("filter", len(grouped), frequent)

        ordered = frequent.sort_values(
            ["family_size", "normalized_agreement_title"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(frequent), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
