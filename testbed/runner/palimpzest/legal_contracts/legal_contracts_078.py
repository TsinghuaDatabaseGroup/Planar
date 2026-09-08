#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-078."""

from __future__ import annotations

import ast
import json
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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-078"
DATASET = "contract-exhibit"


def normalize_string_list(value) -> list[str]:
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return []
    parsed = value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value.strip())
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
    if pd.api.types.is_list_like(parsed) and not isinstance(parsed, dict):
        parsed = list(parsed)
    else:
        parsed = [parsed]
    return [
        normalized
        for item in parsed
        if (normalized := " ".join(str(item).strip().split()))
    ]


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        long_documents = documents.loc[
            documents["word_count"] > 5000
        ].reset_index(drop=True)
        tracker.record("filter", len(documents), long_documents)

        policy_plan = memory_dataset(TASK_ID, long_documents).sem_filter(
            filter=(
                "Keep the document only if it is a substantive insider-trading "
                "policy."
            ),
            depends_on=["text"],
        )
        started = time.time()
        policy_result = policy_plan.run(config)
        policies = result_frame(policy_result, long_documents)
        tracker.record_semantic(
            "sem_filter", len(long_documents), policies, policy_result,
            time.time() - started,
        )

        coverage_plan = memory_dataset(
            f"{TASK_ID}-coverage", policies
        ).sem_filter(
            filter=(
                "Keep the policy only if it discusses Rule 10b5-1 plans and "
                "expressly covers family or household members of covered persons."
            ),
            depends_on=["text"],
        )
        started = time.time()
        coverage_result = coverage_plan.run(config)
        covered = result_frame(coverage_result, policies)
        tracker.record_semantic(
            "sem_filter", len(policies), covered, coverage_result,
            time.time() - started,
        )

        date_plan = memory_dataset(f"{TASK_ID}-date", covered).sem_filter(
            filter=(
                "Keep the policy only if it states no policy approval or "
                "adoption date. A general effective date is not an approval or "
                "adoption date."
            ),
            depends_on=["text"],
        )
        started = time.time()
        date_result = date_plan.run(config)
        undated = result_frame(date_result, covered)
        tracker.record_semantic(
            "sem_filter", len(covered), undated, date_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", undated
        ).sem_map(
            cols=[
                {
                    "name": "company_names",
                    "type": list[str],
                    "desc": (
                        "All company names stated in the policy, excluding "
                        "individual officer names."
                    ),
                }
            ],
            desc="Extract all stated company names.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result, undated, ["company_names"])
        extracted["company_names"] = extracted["company_names"].map(
            normalize_string_list
        )
        extracted = extracted[
            ["document_id", "company_names"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(undated), extracted, extraction_result,
            time.time() - started,
        )

        answer = df_records(extracted)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
