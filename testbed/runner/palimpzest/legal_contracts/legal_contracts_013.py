#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-013."""

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

TASK_ID = "legal_contracts-013"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


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

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the agreement only if it contains a contractual "
                "acquisition-or-control standstill. Exclude insider-trading "
                "rules, statutory anti-takeover descriptions, securities-transfer "
                "lock-ups, instrument-exercise blockers, and no-shop duties."
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
                    "desc": "A concise stated agreement type.",
                },
                {
                    "name": "standstill_duration_years",
                    "type": float | None,
                    "desc": (
                        "The contractual acquisition-or-control standstill "
                        "duration normalized to years, or null when unstated or "
                        "unquantifiable."
                    ),
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": "The expressly stated governing-law jurisdiction.",
                },
            ],
            desc="Extract agreement type, standstill duration, and governing law.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["agreement_type", "standstill_duration_years", "governing_law"]
        extracted = result_frame(extraction_result, agreements, generated)
        extracted["agreement_type"] = extracted["agreement_type"].map(normalize_text)
        extracted["standstill_duration_years"] = extracted[
            "standstill_duration_years"
        ].map(normalize_number)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(agreements), extracted, extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["standstill_duration_years"].notna()
            & (extracted["standstill_duration_years"] >= 2),
            ["document_id", *generated],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
