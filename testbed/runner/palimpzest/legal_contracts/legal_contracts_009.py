#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-009."""

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
    normalize_enum,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-009"
DATASET = "contract-exhibit"
COVERAGE_GROUPS = ("beyond_sec_minimum", "sec_minimum_only")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        policy_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is an EX-97 clawback or "
                "compensation recoupment policy."
            ),
            depends_on=["text"],
        )
        started = time.time()
        policy_result = policy_plan.run(config)
        policies = result_frame(policy_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            policies,
            policy_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", policies
        ).sem_map(
            cols=[
                {
                    "name": "coverage_group",
                    "type": str | None,
                    "desc": (
                        "Exactly beyond_sec_minimum if recovery extends beyond "
                        "SEC minimum requirements; otherwise sec_minimum_only."
                    ),
                },
                {
                    "name": "misconduct_trigger",
                    "type": bool,
                    "desc": (
                        "True if employee misconduct is a recovery trigger "
                        "separate from an accounting restatement; false otherwise."
                    ),
                },
            ],
            desc=(
                "Extract the recovery-coverage group and whether employee "
                "misconduct is a separate trigger."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            policies,
            ["coverage_group", "misconduct_trigger"],
        )
        extracted["coverage_group"] = extracted["coverage_group"].map(
            lambda value: normalize_enum(value, COVERAGE_GROUPS)
        )
        extracted["misconduct_trigger"] = extracted["misconduct_trigger"].map(
            parse_bool
        )
        tracker.record_semantic(
            "sem_map",
            len(policies),
            extracted,
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("coverage_group", sort=False, dropna=False)
            .agg(
                policy_count=("document_id", "size"),
                misconduct_fraction=("misconduct_trigger", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["misconduct_fraction"] = grouped[
                "misconduct_fraction"
            ].round(4)
        order = {
            group: index for index, group in enumerate(COVERAGE_GROUPS)
        }
        grouped = grouped.sort_values(
            "coverage_group",
            key=lambda values: values.map(order),
        ).reset_index(drop=True)
        tracker.record("groupby", len(extracted), grouped)

        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
