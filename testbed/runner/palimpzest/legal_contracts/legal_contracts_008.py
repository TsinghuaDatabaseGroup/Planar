#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-008."""

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

TASK_ID = "legal_contracts-008"
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

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the agreement only if it contains an operative non-compete "
                "restriction rather than an incidental reference to non-competition."
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

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", agreements
        ).sem_map(
            cols=[
                {
                    "name": "document_name",
                    "type": str | None,
                    "desc": "The stated name or title of the agreement.",
                },
                {
                    "name": "non_compete_months",
                    "type": float | None,
                    "desc": (
                        "The operative non-compete duration normalized to months, "
                        "or null when unstated or unquantifiable."
                    ),
                },
                {
                    "name": "geographic_scope",
                    "type": str | None,
                    "desc": (
                        "The geographic scope stated for the non-compete "
                        "restriction, or null when unstated."
                    ),
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": (
                        "The expressly stated governing-law jurisdiction, or "
                        "null when unstated."
                    ),
                },
            ],
            desc=(
                "Extract the document name, non-compete duration, geographic "
                "scope, and governing law."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "document_name",
            "non_compete_months",
            "geographic_scope",
            "governing_law",
        ]
        extracted = result_frame(extraction_result, agreements, generated)
        extracted["document_name"] = extracted["document_name"].map(normalize_text)
        extracted["non_compete_months"] = extracted[
            "non_compete_months"
        ].map(normalize_number)
        for column in ("geographic_scope", "governing_law"):
            extracted[column] = extracted[column].map(normalize_text)
        tracker.record_semantic(
            "sem_map",
            len(agreements),
            extracted,
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["non_compete_months"].notna()
            & (extracted["non_compete_months"] <= 6)
            & extracted["geographic_scope"].notna()
            & extracted["governing_law"].notna(),
            generated,
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        answer = df_records(qualifying)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
