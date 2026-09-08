#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-083."""

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
    normalize_scalar_value,
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-083"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def sorted_distinct_rounded(values: pd.Series) -> list[float]:
    return sorted({round(float(value), 4) for value in values if pd.notna(value)})


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a standalone non-disclosure "
                "agreement and is not also an EX-10 employment or restrictive-"
                "covenant agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        exception_plan = memory_dataset(
            f"{TASK_ID}-six-exceptions", scoped
        ).sem_filter(
            filter=(
                "Keep the agreement only if it includes all six operative "
                "confidentiality exceptions: public information, prior knowledge, "
                "independent development, lawful unrestricted third-party receipt, "
                "legal compulsion, and disclosure with consent."
            ),
            depends_on=["text"],
        )
        started = time.time()
        exception_result = exception_plan.run(config)
        complete_exceptions = result_frame(exception_result, scoped)
        tracker.record_semantic(
            "sem_filter", len(scoped), complete_exceptions, exception_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", complete_exceptions
        ).sem_map(
            cols=[
                {
                    "name": "term_years",
                    "type": float | None,
                    "desc": (
                        "The finite confidentiality term normalized to years, or null "
                        "when absent, perpetual, or not quantifiable."
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
            desc="Extract the finite confidentiality term and stated governing law.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result, complete_exceptions, ["term_years", "governing_law"]
        )
        extracted["term_years"] = extracted["term_years"].map(normalize_number)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(complete_exceptions), extracted, extraction_result,
            time.time() - started,
        )

        complete = extracted.loc[
            extracted["term_years"].notna() & extracted["governing_law"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), complete)

        grouped = (
            complete.groupby("governing_law", sort=False)
            .agg(
                agreement_count=("document_id", "size"),
                term_lengths_years=("term_years", sorted_distinct_rounded),
            )
            .reset_index()
        )
        tracker.record("groupby", len(complete), grouped)

        frequent = grouped.loc[grouped["agreement_count"] >= 3].reset_index(drop=True)
        tracker.record("filter", len(grouped), frequent)

        ordered = frequent.sort_values(
            ["agreement_count", "governing_law"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(frequent), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
