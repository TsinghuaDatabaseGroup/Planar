#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-070."""

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
    normalize_text_value,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-070"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


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

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure agreement, "
                "a unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement that contains an intellectual-property "
                "ownership clause."
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
                    "name": "residual_information_present",
                    "type": bool,
                    "desc": (
                        "True if the agreement grants a residual-information right; "
                        "otherwise false."
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
                {
                    "name": "return_of_materials",
                    "type": bool,
                    "desc": (
                        "True if return or destruction of confidential materials is "
                        "required; otherwise false."
                    ),
                },
                {
                    "name": "injunctive_relief",
                    "type": bool,
                    "desc": (
                        "True if the agreement provides injunctive or equitable "
                        "relief; otherwise false."
                    ),
                },
            ],
            desc=(
                "Extract the residual-information right, governing law, return-or-"
                "destroy requirement, and injunctive-relief provision."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "residual_information_present",
            "governing_law",
            "return_of_materials",
            "injunctive_relief",
        ]
        extracted = result_frame(extraction_result, scoped, generated)
        for column in (
            "residual_information_present",
            "return_of_materials",
            "injunctive_relief",
        ):
            extracted[column] = extracted[column].map(parse_bool)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(scoped), extracted, extraction_result, time.time() - started
        )

        projected = extracted[
            [
                "residual_information_present",
                "governing_law",
                "return_of_materials",
                "injunctive_relief",
            ]
        ].rename(
            columns={
                "return_of_materials": "return_flag",
                "injunctive_relief": "injunctive_flag",
            }
        )
        projected["return_flag"] = projected["return_flag"].astype(int)
        projected["injunctive_flag"] = projected["injunctive_flag"].astype(int)
        projected = projected.reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("residual_information_present", sort=False)
            .agg(
                agreement_count=("residual_information_present", "size"),
                most_common_governing_law=("governing_law", stable_mode_optional),
                return_of_materials_rate=("return_flag", "mean"),
                injunctive_relief_rate=("injunctive_flag", "mean"),
            )
            .reset_index()
        )
        grouped["return_of_materials_rate"] = grouped[
            "return_of_materials_rate"
        ].round(4)
        grouped["injunctive_relief_rate"] = grouped[
            "injunctive_relief_rate"
        ].round(4)
        tracker.record("groupby", len(projected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
