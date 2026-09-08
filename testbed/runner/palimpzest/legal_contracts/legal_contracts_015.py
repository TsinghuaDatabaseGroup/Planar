#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-015."""

from __future__ import annotations

import os
import sys
import time

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

TASK_ID = "legal_contracts-015"
DATASET = "contract-exhibit"
DIRECTIONS = ("mutual", "unilateral", "unclear")


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
                {
                    "name": "has_non_use_restriction",
                    "type": bool,
                    "desc": (
                        "True if an explicit restriction on use of protected "
                        "information is present; false otherwise."
                    ),
                },
            ],
            desc="Extract nondisclosure direction and the non-use indicator.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            agreements,
            ["non_disclosure_direction", "has_non_use_restriction"],
        )
        extracted["non_disclosure_direction"] = extracted[
            "non_disclosure_direction"
        ].map(lambda value: normalize_enum(value, DIRECTIONS))
        extracted["has_non_use_restriction"] = extracted[
            "has_non_use_restriction"
        ].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(agreements), extracted, extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby(
                "non_disclosure_direction", sort=False, dropna=False
            )
            .agg(
                agreement_count=("document_id", "size"),
                non_use_percentage=("has_non_use_restriction", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["non_use_percentage"] = (
                grouped["non_use_percentage"] * 100.0
            ).round(2)
        tracker.record("groupby", len(extracted), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
