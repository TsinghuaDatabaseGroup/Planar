#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-046."""

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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-046"
DATASET = "contract-exhibit"
DIRECTIONS = ("mutual", "unilateral")


def normalize_term_years(value):
    value = normalize_scalar_value(value)
    normalized = str(value).strip().lower()
    if "perpetual" in normalized or "indefinite" in normalized:
        return "perpetual"
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
                "Keep the document only if it is a non-disclosure or "
                "confidentiality agreement expressly governed by Delaware law."
            ),
            depends_on=["text"],
        )
        started = time.time()
        agreement_result = agreement_plan.run(config)
        agreements = result_frame(agreement_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            agreements,
            agreement_result,
            time.time() - started,
        )

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if its confidentiality term is at "
                "least three years or perpetual or indefinite and it includes "
                "all six operative confidentiality carve-outs for publicly "
                "available information, prior knowledge, independent development, "
                "lawful unrestricted third-party receipt, legally compelled "
                "disclosure, and disclosure authorized by the disclosing party."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, agreements)
        tracker.record_semantic(
            "sem_filter",
            len(agreements),
            qualifying,
            condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {
                    "name": "term_duration_years",
                    "type": str,
                    "desc": (
                        "The confidentiality term as a number of years for a fixed "
                        "duration, or exactly perpetual for a perpetual or "
                        "indefinite term."
                    ),
                },
                {
                    "name": "non_disclosure_direction",
                    "type": str | None,
                    "desc": (
                        "Exactly mutual if both sides owe the nondisclosure duty, "
                        "or unilateral if only one side owes it."
                    ),
                },
            ],
            desc=(
                "Extract the normalized confidentiality term and nondisclosure "
                "direction."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["term_duration_years", "non_disclosure_direction"]
        extracted = result_frame(extraction_result, qualifying, generated)
        extracted["term_duration_years"] = extracted[
            "term_duration_years"
        ].map(normalize_term_years)
        extracted["non_disclosure_direction"] = extracted[
            "non_disclosure_direction"
        ].map(lambda value: normalize_enum(value, DIRECTIONS))
        extracted = extracted[
            ["document_id", *generated]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(qualifying),
            extracted,
            extraction_result,
            time.time() - started,
        )

        answer = df_records(extracted)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
