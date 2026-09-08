#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-072."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-072"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_iso_date(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is an insider-trading policy that "
                "explicitly covers family or household members of covered persons."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", scoped
        ).sem_map(
            cols=[
                {
                    "name": "document_name",
                    "type": str,
                    "desc": "The stated document name.",
                },
                {
                    "name": "policy_title",
                    "type": str,
                    "desc": "The stated policy title.",
                },
                {
                    "name": "discusses_10b5_1",
                    "type": bool,
                    "desc": (
                        "True if the policy discusses Rule 10b5-1 plans; otherwise false."
                    ),
                },
                {
                    "name": "prohibits_short_sales",
                    "type": bool,
                    "desc": "True if the policy prohibits short sales; otherwise false.",
                },
                {
                    "name": "prohibits_pledging",
                    "type": bool,
                    "desc": "True if the policy prohibits pledging; otherwise false.",
                },
                {
                    "name": "policy_date",
                    "type": str | None,
                    "desc": (
                        "The expressly stated policy approval or effective date in "
                        "YYYY-MM-DD form, or null when neither is stated."
                    ),
                },
            ],
            desc=(
                "Extract the policy name and title, three policy features, and its "
                "stated approval or effective date."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "document_name",
            "policy_title",
            "discusses_10b5_1",
            "prohibits_short_sales",
            "prohibits_pledging",
            "policy_date",
        ]
        extracted = result_frame(extraction_result, scoped, generated)
        for column in ("document_name", "policy_title"):
            extracted[column] = extracted[column].map(normalize_text)
        for column in (
            "discusses_10b5_1",
            "prohibits_short_sales",
            "prohibits_pledging",
        ):
            extracted[column] = extracted[column].map(parse_bool)
        extracted["policy_date"] = extracted["policy_date"].map(normalize_iso_date)
        tracker.record_semantic(
            "sem_map", len(scoped), extracted, extraction_result, time.time() - started
        )

        qualifying = extracted.loc[
            extracted["discusses_10b5_1"].eq(True)
            & extracted["prohibits_short_sales"].eq(True)
            & extracted["prohibits_pledging"].eq(True)
            & extracted["policy_date"].notna()
            & extracted["policy_date"].lt("2024-01-01"),
            ["document_name", "policy_title", "policy_date"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        ordered = qualifying.sort_values(
            ["policy_date", "document_name"],
            ascending=[True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(qualifying), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
