#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-061."""

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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-061"
DATASET = "contract-exhibit"
AGREEMENT_TYPES = (
    "mutual_nda",
    "unilateral_nda",
    "confidentiality_standstill",
    "not_in_scope",
)
DIRECTIONS = ("mutual", "unilateral", "not_in_scope")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is categorized as a mutual "
                "non-disclosure agreement, unilateral non-disclosure agreement, "
                "or confidentiality-and-standstill agreement."
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
                    "name": "agreement_type",
                    "type": str | None,
                    "desc": (
                        "Exactly mutual_nda, unilateral_nda, or "
                        "confidentiality_standstill; use not_in_scope only for a "
                        "document outside those categories."
                    ),
                },
                {
                    "name": "non_disclosure_direction",
                    "type": str | None,
                    "desc": (
                        "Exactly mutual or unilateral according to the operative "
                        "nondisclosure duty; use not_in_scope only for an out-of-"
                        "scope document."
                    ),
                },
            ],
            desc="Extract the agreement category and operative nondisclosure direction.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["agreement_type", "non_disclosure_direction"]
        extracted = result_frame(extraction_result, agreements, generated)
        extracted["agreement_type"] = extracted["agreement_type"].map(
            lambda value: normalize_enum(value, AGREEMENT_TYPES)
        )
        extracted["non_disclosure_direction"] = extracted[
            "non_disclosure_direction"
        ].map(lambda value: normalize_enum(value, DIRECTIONS))
        tracker.record_semantic(
            "sem_map", len(agreements), extracted, extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby(generated, sort=False, dropna=False)
            .size()
            .rename("document_count")
            .reset_index()
        )
        tracker.record("groupby", len(extracted), grouped)

        ordered = grouped.sort_values(
            ["document_count", "agreement_type", "non_disclosure_direction"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
