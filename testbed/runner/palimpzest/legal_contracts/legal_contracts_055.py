#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-055."""

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

TASK_ID = "legal_contracts-055"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        residual_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter="Keep the agreement only if it contains a residual-information clause.",
            depends_on=["text"],
        )
        started = time.time()
        residual_result = residual_plan.run(config)
        residual_agreements = result_frame(residual_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), residual_agreements, residual_result,
            time.time() - started,
        )

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", residual_agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it expressly states a governing "
                "law and contains an intellectual-property or work-product "
                "ownership clause."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, residual_agreements)
        tracker.record_semantic(
            "sem_filter", len(residual_agreements), qualifying, condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {
                    "name": "exhibit_type",
                    "type": str | None,
                    "desc": (
                        "Use NDA for an NDA, the applicable EX-xx label for an "
                        "exhibit, and NDA|EX-xx for a dual-status document."
                    ),
                },
                {
                    "name": "agreement_type",
                    "type": str | None,
                    "desc": "A concise stated agreement type.",
                },
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": "The expressly stated governing-law jurisdiction.",
                },
            ],
            desc="Extract exhibit type, agreement type, and governing law.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["exhibit_type", "agreement_type", "governing_law"]
        extracted = result_frame(extraction_result, qualifying, generated)
        for column in generated:
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
