#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-071."""

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

TASK_ID = "legal_contracts-071"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_direction(value) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().casefold()
    ).strip("_")
    return normalized if normalized in {"mutual", "unilateral", "unclear"} else "unclear"


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def compare_durations(row: pd.Series) -> str:
    confidentiality = row["confidentiality_term_years"]
    noncompete = row["noncompete_duration_years"]
    if confidentiality == noncompete:
        return "equal"
    if confidentiality > noncompete:
        return "confidentiality_term_longer"
    return "noncompete_duration_longer"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is both a non-disclosure agreement "
                "and an employment or restrictive-covenant agreement, corresponding "
                "to dual NDA|EX-10 status."
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
                    "name": "nda_direction",
                    "type": str,
                    "desc": (
                        "Exactly mutual, unilateral, or unclear according to the "
                        "NDA nondisclosure obligations."
                    ),
                },
                {
                    "name": "confidentiality_term_years",
                    "type": float | None,
                    "desc": (
                        "The finite confidentiality term normalized to years, or null "
                        "when unstated, perpetual, or not quantifiable."
                    ),
                },
                {
                    "name": "noncompete_duration_years",
                    "type": float | None,
                    "desc": (
                        "The finite non-compete duration normalized to years, or null "
                        "when unstated or not quantifiable."
                    ),
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The expressly stated governing-law jurisdiction, or null "
                        "when none is stated."
                    ),
                },
            ],
            desc=(
                "Extract NDA direction, finite confidentiality and non-compete "
                "durations, and expressly stated governing law."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "nda_direction",
            "confidentiality_term_years",
            "noncompete_duration_years",
            "governing_law",
        ]
        extracted = result_frame(extraction_result, scoped, generated)
        extracted["nda_direction"] = extracted["nda_direction"].map(
            normalize_direction
        )
        for column in ("confidentiality_term_years", "noncompete_duration_years"):
            extracted[column] = extracted[column].map(normalize_number)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(scoped), extracted, extraction_result, time.time() - started
        )

        complete = extracted.loc[
            extracted["confidentiality_term_years"].notna()
            & extracted["noncompete_duration_years"].notna()
            & extracted["governing_law"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), complete)

        projected = complete[
            [
                "document_id",
                "nda_direction",
                "confidentiality_term_years",
                "noncompete_duration_years",
                "governing_law",
            ]
        ].copy()
        projected["duration_comparison"] = projected.apply(compare_durations, axis=1)
        projected = projected.reset_index(drop=True)
        tracker.record("project", len(complete), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
