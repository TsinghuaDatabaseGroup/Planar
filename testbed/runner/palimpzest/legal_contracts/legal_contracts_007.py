#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-007."""

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

TASK_ID = "legal_contracts-007"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        long_documents = documents.loc[
            documents["word_count"] > 7500
        ].reset_index(drop=True)
        tracker.record("filter", len(documents), long_documents)

        policy_plan = memory_dataset(TASK_ID, long_documents).sem_filter(
            filter=(
                "Keep the document only if it is a substantive insider-trading "
                "policy and explicitly extends its trading restrictions to "
                "family or household members of covered persons."
            ),
            depends_on=["text"],
        )
        started = time.time()
        policy_result = policy_plan.run(config)
        policies = result_frame(policy_result, long_documents)
        tracker.record_semantic(
            "sem_filter",
            len(long_documents),
            policies,
            policy_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", policies
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str | None,
                    "desc": "The company name expressly stated in the policy.",
                }
            ],
            desc="Extract the stated company name.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result, policies, ["company_name"]
        )
        extracted["company_name"] = extracted["company_name"].map(
            normalize_text
        )
        extracted = extracted[
            ["document_id", "company_name"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(policies),
            extracted,
            extraction_result,
            time.time() - started,
        )

        answer = df_records(extracted)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
