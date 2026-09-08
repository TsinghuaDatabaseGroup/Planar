#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-068."""

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
    normalize_text_value,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-068"
DATASET = "contract-exhibit"
EXCEPTION_COLUMNS = (
    "public_information", "prior_knowledge", "independent_development",
    "lawful_third_party_receipt", "legal_compulsion", "consent",
)


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_title(value) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


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
                "agreement title."
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
                {"name": "agreement_title", "type": str | None,
                 "desc": "The identifiable stated agreement title."},
                *[
                    {"name": column, "type": bool,
                     "desc": f"True if the operative {column.replace('_', ' ')} confidentiality carve-out is present; false otherwise."}
                    for column in EXCEPTION_COLUMNS
                ],
                {"name": "return_of_materials", "type": bool,
                 "desc": "True if return or destruction of confidential materials is required; false otherwise."},
                {"name": "injunctive_relief", "type": bool,
                 "desc": "True if injunctive or equitable relief is provided; false otherwise."},
            ],
            desc=(
                "Extract agreement title, six carve-out indicators, return-or-"
                "destroy language, and injunctive relief."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["agreement_title", *EXCEPTION_COLUMNS, "return_of_materials", "injunctive_relief"]
        extracted = result_frame(extraction_result, agreements, generated)
        extracted["agreement_title"] = extracted["agreement_title"].map(normalize_text)
        for column in (*EXCEPTION_COLUMNS, "return_of_materials", "injunctive_relief"):
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(agreements), extracted, extraction_result,
            time.time() - started,
        )

        projected = pd.DataFrame(index=extracted.index)
        projected["normalized_agreement_title"] = extracted["agreement_title"].map(
            normalize_title
        )
        projected["exception_count"] = extracted[list(EXCEPTION_COLUMNS)].sum(axis=1)
        projected["has_return_and_injunctive"] = (
            extracted["return_of_materials"] & extracted["injunctive_relief"]
        ).astype(int)
        projected["has_all_six"] = (projected["exception_count"] == 6).astype(int)
        projected = projected.reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("normalized_agreement_title", sort=False)
            .agg(
                family_size=("normalized_agreement_title", "size"),
                avg_exception_count=("exception_count", "mean"),
                return_and_injunctive_share=("has_return_and_injunctive", "mean"),
                all_six_exceptions_share=("has_all_six", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_exception_count"] = grouped["avg_exception_count"].round(2)
            grouped["return_and_injunctive_share"] = grouped[
                "return_and_injunctive_share"
            ].round(4)
            grouped["all_six_exceptions_share"] = grouped[
                "all_six_exceptions_share"
            ].round(4)
        tracker.record("groupby", len(projected), grouped)

        large_families = grouped.loc[
            grouped["family_size"] >= 5
        ].reset_index(drop=True)
        tracker.record("filter", len(grouped), large_families)

        ordered = large_families.sort_values(
            ["family_size", "normalized_agreement_title"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(large_families), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
