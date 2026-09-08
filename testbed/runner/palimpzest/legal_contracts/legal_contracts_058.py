#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-058."""

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
    normalize_enum,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-058"
DATASET = "contract-exhibit"
DIRECTIONS = ("mutual", "unilateral", "unclear")
EXCEPTION_COLUMNS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "lawful_third_party_receipt",
    "legally_compelled_disclosure",
    "disclosure_with_consent",
)


def recognized_exceptions(row) -> tuple[str, ...]:
    return tuple(column for column in EXCEPTION_COLUMNS if row[column])


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
                {
                    "name": "non_disclosure_direction",
                    "type": str | None,
                    "desc": (
                        "Exactly mutual, unilateral, or unclear according to "
                        "who owes the nondisclosure duty."
                    ),
                },
                *[
                    {
                        "name": column,
                        "type": bool,
                        "desc": (
                            f"True if the operative {column.replace('_', ' ')} "
                            "confidentiality carve-out is present; false otherwise."
                        ),
                    }
                    for column in EXCEPTION_COLUMNS
                ],
            ],
            desc=(
                "Extract nondisclosure direction and the presence of each of "
                "the six standard confidentiality carve-outs."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["non_disclosure_direction", *EXCEPTION_COLUMNS]
        extracted = result_frame(extraction_result, agreements, generated)
        extracted["non_disclosure_direction"] = extracted[
            "non_disclosure_direction"
        ].map(lambda value: normalize_enum(value, DIRECTIONS))
        for column in EXCEPTION_COLUMNS:
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(agreements), extracted, extraction_result,
            time.time() - started,
        )

        projected = extracted[["non_disclosure_direction"]].copy()
        projected["recognized_exceptions"] = pd.Series(
            (recognized_exceptions(row) for _, row in extracted.iterrows()),
            index=extracted.index,
            dtype="object",
        )
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby(
                ["non_disclosure_direction", "recognized_exceptions"],
                sort=False,
                dropna=False,
            )
            .size()
            .rename("agreement_count")
            .reset_index()
        )
        tracker.record("groupby", len(projected), grouped)

        ordered = grouped.sort_values(
            ["agreement_count", "non_disclosure_direction", "recognized_exceptions"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        ordered["recognized_exceptions"] = ordered[
            "recognized_exceptions"
        ].map(list)
        tracker.record("orderby", len(grouped), ordered)

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
