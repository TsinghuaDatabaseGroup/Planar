#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-041."""

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

TASK_ID = "legal_contracts-041"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def parse_optional_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
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

        policy_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a substantive insider-trading "
                "policy."
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
                    "name": "covered_persons",
                    "type": str | None,
                    "desc": (
                        "A concise description of the categories of persons "
                        "covered by the trading restrictions."
                    ),
                },
                {
                    "name": "extends_to_family",
                    "type": bool | None,
                    "desc": (
                        "True only if the trading restrictions explicitly extend "
                        "to covered persons' family or household members; false "
                        "otherwise."
                    ),
                },
            ],
            desc=(
                "Extract covered-person categories and whether restrictions "
                "explicitly extend to family or household members."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            policies,
            ["covered_persons", "extends_to_family"],
        )
        extracted["covered_persons"] = extracted["covered_persons"].map(
            normalize_text
        )
        extracted["extends_to_family"] = extracted["extends_to_family"].map(
            parse_optional_bool
        )
        tracker.record_semantic(
            "sem_map",
            len(policies),
            extracted,
            extraction_result,
            time.time() - started,
        )

        without_family = extracted.loc[
            extracted["extends_to_family"].eq(False)
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), without_family)

        projected = without_family[
            ["document_id", "covered_persons"]
        ].reset_index(drop=True)
        tracker.record("project", len(without_family), projected)

        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
