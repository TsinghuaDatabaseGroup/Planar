#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-059."""

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

TASK_ID = "legal_contracts-059"
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
    return sorted(
        {
            normalized
            for item in parsed
            if (normalized := " ".join(str(item).strip().split()))
        }
    )


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        certification_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a genuine Sarbanes-Oxley "
                "Section 906 certification."
            ),
            depends_on=["text"],
        )
        started = time.time()
        certification_result = certification_plan.run(config)
        certifications = result_frame(certification_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), certifications, certification_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", certifications
        ).sem_map(
            cols=[
                {
                    "name": "certifying_officers",
                    "type": list[str],
                    "desc": (
                        "A deduplicated list of the full names of certifying "
                        "officers named as signatories."
                    ),
                }
            ],
            desc="Extract the distinct certifying officers.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result, certifications, ["certifying_officers"]
        )
        extracted["certifying_officers"] = extracted[
            "certifying_officers"
        ].map(normalize_string_list)
        tracker.record_semantic(
            "sem_map", len(certifications), extracted, extraction_result,
            time.time() - started,
        )

        projected = extracted[["certifying_officers"]].copy()
        projected["officer_count"] = projected["certifying_officers"].map(len)
        projected = projected[["officer_count"]].reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("officer_count", sort=False)
            .size()
            .rename("certification_count")
            .reset_index()
        )
        tracker.record("groupby", len(projected), grouped)

        ordered = grouped.sort_values("officer_count", kind="stable").reset_index(
            drop=True
        )
        tracker.record("orderby", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
