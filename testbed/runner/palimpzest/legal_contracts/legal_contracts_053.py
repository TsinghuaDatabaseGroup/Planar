#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-053."""

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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-053"
DATASET = "contract-exhibit"


def normalize_iso_date(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        policy_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter="Keep the document only if it is an insider-trading policy.",
            depends_on=["text"],
        )
        started = time.time()
        policy_result = policy_plan.run(config)
        policies = result_frame(policy_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), policies, policy_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", policies
        ).sem_map(
            cols=[
                {
                    "name": "approval_date",
                    "type": str | None,
                    "desc": (
                        "The policy approval or adoption date explicitly stated "
                        "in the document, formatted as YYYY-MM-DD; null when none "
                        "is stated. Do not substitute a general effective date."
                    ),
                }
            ],
            desc="Extract the explicitly stated policy approval or adoption date.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result, policies, ["approval_date"])
        extracted["approval_date"] = extracted["approval_date"].map(normalize_iso_date)
        extracted = extracted[
            ["document_id", "approval_date"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(policies), extracted, extraction_result,
            time.time() - started,
        )

        dated = extracted.loc[
            extracted["approval_date"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), dated)

        ordered = dated.sort_values(
            ["approval_date", "document_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(dated), ordered)

        limited = ordered.head(1).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        rows = df_records(limited)
        answer = rows[0] if rows else {}

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
