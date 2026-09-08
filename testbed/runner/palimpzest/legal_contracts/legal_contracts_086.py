#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-086."""

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
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-086"
DATASET = "contract-exhibit"


def normalize_governing_law(value) -> str | None:
    """Normalize null-like extraction output without inferring a jurisdiction."""
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is classified as a non-disclosure "
                "or confidentiality agreement, including a dual-labelled NDA "
                "exhibit, and its expressly stated effective date is strictly "
                "after 2010-01-01."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            scoped,
            scope_result,
            time.time() - started,
        )

        electronic_plan = memory_dataset(
            f"{TASK_ID}-electronic-definition",
            scoped,
        ).sem_filter(
            filter=(
                "Keep the agreement only if its definition of confidential "
                "information explicitly includes electronic data or records, "
                "email, software or code, metadata, or other computer-readable "
                "material."
            ),
            depends_on=["text"],
        )
        started = time.time()
        electronic_result = electronic_plan.run(config)
        electronic = result_frame(electronic_result, scoped)
        tracker.record_semantic(
            "sem_filter",
            len(scoped),
            electronic,
            electronic_result,
            time.time() - started,
        )

        destruction_plan = memory_dataset(
            f"{TASK_ID}-destruction",
            electronic,
        ).sem_filter(
            filter=(
                "Keep the agreement only if it requires destruction of at least "
                "some confidential materials or copies. Exclude clauses that "
                "require only return and clauses that merely allow an "
                "unrestricted choice between return and destruction."
            ),
            depends_on=["text"],
        )
        started = time.time()
        destruction_result = destruction_plan.run(config)
        destruction = result_frame(destruction_result, electronic)
        tracker.record_semantic(
            "sem_filter",
            len(electronic),
            destruction,
            destruction_result,
            time.time() - started,
        )

        survival_plan = memory_dataset(
            f"{TASK_ID}-survival",
            destruction,
        ).sem_filter(
            filter=(
                "Keep the agreement only if confidentiality or non-use duties "
                "expressly survive termination or expiry, or expressly continue "
                "beyond the stated agreement term."
            ),
            depends_on=["text"],
        )
        started = time.time()
        survival_result = survival_plan.run(config)
        surviving = result_frame(survival_result, destruction)
        tracker.record_semantic(
            "sem_filter",
            len(destruction),
            surviving,
            survival_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-governing-law",
            surviving,
        ).sem_map(
            cols=[
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The concise canonical jurisdiction expressly named in "
                        "the governing-law clause, or null if no governing-law "
                        "jurisdiction is expressly stated. Do not infer it from "
                        "party addresses, court venue, or other locations."
                    ),
                }
            ],
            desc=(
                "Extract and canonicalize the expressly stated governing-law "
                "jurisdiction."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            surviving,
            ["governing_law"],
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_governing_law
        )
        tracker.record_semantic(
            "sem_map",
            len(surviving),
            extracted,
            extraction_result,
            time.time() - started,
        )

        with_governing_law = extracted.loc[
            extracted["governing_law"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), with_governing_law)

        grouped = (
            with_governing_law.groupby(
                "governing_law",
                as_index=False,
                dropna=False,
            )
            .size()
            .rename(columns={"size": "matching_count"})
        )
        tracker.record("groupby", len(with_governing_law), grouped)

        ordered = grouped.sort_values(
            ["matching_count", "governing_law"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)
        answer = df_records(ordered[["governing_law", "matching_count"]])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
