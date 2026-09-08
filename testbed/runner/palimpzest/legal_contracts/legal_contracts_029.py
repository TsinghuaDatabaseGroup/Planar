#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-029."""

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

TASK_ID = "legal_contracts-029"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        non_compete_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the agreement only if it contains an operative "
                "non-compete restriction."
            ),
            depends_on=["text"],
        )
        started = time.time()
        non_compete_result = non_compete_plan.run(config)
        non_compete_agreements = result_frame(non_compete_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            non_compete_agreements,
            non_compete_result,
            time.time() - started,
        )

        standstill_plan = memory_dataset(
            f"{TASK_ID}-standstill", non_compete_agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains no operative "
                "standstill restriction."
            ),
            depends_on=["text"],
        )
        started = time.time()
        standstill_result = standstill_plan.run(config)
        no_standstill = result_frame(
            standstill_result, non_compete_agreements
        )
        tracker.record_semantic(
            "sem_filter",
            len(non_compete_agreements),
            no_standstill,
            standstill_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", no_standstill
        ).sem_map(
            cols=[
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The expressly stated governing-law jurisdiction, or "
                        "null when unstated."
                    ),
                },
                {
                    "name": "non_compete_months",
                    "type": float | None,
                    "desc": (
                        "The operative non-compete duration normalized to months, "
                        "or null when unstated or unquantifiable."
                    ),
                },
            ],
            desc="Extract the governing law and normalized non-compete duration.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            no_standstill,
            ["governing_law", "non_compete_months"],
        )
        extracted["governing_law"] = extracted["governing_law"].map(
            normalize_text
        )
        extracted["non_compete_months"] = extracted[
            "non_compete_months"
        ].map(normalize_number)
        tracker.record_semantic(
            "sem_map",
            len(no_standstill),
            extracted,
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["governing_law"].notna()
            & extracted["non_compete_months"].notna()
            & (extracted["non_compete_months"] <= 6),
            ["document_id", "governing_law", "non_compete_months"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
