#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-057."""

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

TASK_ID = "legal_contracts-057"
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
                "Keep the document only if it is an employment agreement or a "
                "standalone restrictive-covenant agreement."
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

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains an operative non-compete "
                "restriction lasting at least four years and expressly states "
                "its geographic scope."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, agreements)
        tracker.record_semantic(
            "sem_filter", len(agreements), qualifying, condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {
                    "name": "duration_years",
                    "type": float | None,
                    "desc": "The operative non-compete duration normalized to years.",
                },
                {
                    "name": "geographic_scope",
                    "type": str | None,
                    "desc": "The expressly stated geographic scope of the non-compete.",
                },
                {
                    "name": "prohibited_activity_scope",
                    "type": str | None,
                    "desc": (
                        "A concise description of the activities prohibited by "
                        "the non-compete."
                    ),
                },
            ],
            desc=(
                "Extract the non-compete duration, geographic scope, and "
                "prohibited-activity scope."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["duration_years", "geographic_scope", "prohibited_activity_scope"]
        extracted = result_frame(extraction_result, qualifying, generated)
        extracted["duration_years"] = extracted["duration_years"].map(normalize_number)
        for column in ("geographic_scope", "prohibited_activity_scope"):
            extracted[column] = extracted[column].map(normalize_text)
        extracted = extracted[
            ["document_id", *generated]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(qualifying), extracted, extraction_result,
            time.time() - started,
        )

        answer = df_records(extracted)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
