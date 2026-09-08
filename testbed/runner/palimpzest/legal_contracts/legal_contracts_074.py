#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-074."""

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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-074"
DATASET = "contract-exhibit"
FEATURE_COLUMNS = [
    "blackout_period",
    "pre_clearance",
    "rule_10b5_1",
    "family_household_coverage",
    "pledging_prohibited",
]


def normalize_optional_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        if value in {0, 1}:
            return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter="Keep the document only if it is an insider-trading policy.",
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
                    "name": "blackout_period",
                    "type": bool | None,
                    "desc": (
                        "True if the policy has a blackout period, false if it "
                        "definitely does not, or null when undetermined."
                    ),
                },
                {
                    "name": "pre_clearance",
                    "type": bool | None,
                    "desc": (
                        "True if the policy requires pre-clearance, false if it "
                        "definitely does not, or null when undetermined."
                    ),
                },
                {
                    "name": "rule_10b5_1",
                    "type": bool | None,
                    "desc": (
                        "True if the policy discusses Rule 10b5-1 plans, false if it "
                        "definitely does not, or null when undetermined."
                    ),
                },
                {
                    "name": "family_household_coverage",
                    "type": bool | None,
                    "desc": (
                        "True if the policy covers family or household members, false "
                        "if it definitely does not, or null when undetermined."
                    ),
                },
                {
                    "name": "pledging_prohibited",
                    "type": bool | None,
                    "desc": (
                        "True if the policy prohibits pledging, false if it definitely "
                        "does not, or null when undetermined."
                    ),
                },
            ],
            desc="Determine the five requested insider-trading policy features.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result, scoped, FEATURE_COLUMNS)
        for column in FEATURE_COLUMNS:
            extracted[column] = extracted[column].map(normalize_optional_bool)
        tracker.record_semantic(
            "sem_map", len(scoped), extracted, extraction_result, time.time() - started
        )

        complete = extracted.dropna(subset=FEATURE_COLUMNS).copy()
        for column in FEATURE_COLUMNS:
            complete[column] = complete[column].astype(bool)
        complete = complete.reset_index(drop=True)
        tracker.record("filter", len(extracted), complete)

        grouped = (
            complete.groupby(FEATURE_COLUMNS, sort=False)
            .size()
            .rename("policy_count")
            .reset_index()
        )
        tracker.record("groupby", len(complete), grouped)

        ordered = grouped.sort_values(
            ["policy_count", *FEATURE_COLUMNS],
            ascending=[False, True, True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
