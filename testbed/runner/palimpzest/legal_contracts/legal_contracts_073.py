#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-073."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-073"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a clawback or compensation-recovery "
                "policy."
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
                    "name": "lookback_period",
                    "type": str,
                    "desc": "The lookback period stated by the policy.",
                },
                {
                    "name": "restatement_trigger",
                    "type": bool,
                    "desc": (
                        "True if a financial restatement triggers recovery; otherwise false."
                    ),
                },
                {
                    "name": "misconduct_trigger",
                    "type": bool,
                    "desc": "True if misconduct triggers recovery; otherwise false.",
                },
                {
                    "name": "expanded_covered_group",
                    "type": str | None,
                    "desc": (
                        "A concise description of covered persons beyond current or "
                        "former Rule 10D-1 or Section 16 executive officers, such as "
                        "employees, directors, key managers, or other designated senior "
                        "executives; null if coverage does not extend beyond those officers."
                    ),
                },
            ],
            desc=(
                "Extract the policy identity, lookback period, recovery triggers, "
                "and any expanded covered-person group."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "document_name",
            "policy_title",
            "lookback_period",
            "restatement_trigger",
            "misconduct_trigger",
            "expanded_covered_group",
        ]
        extracted = result_frame(extraction_result, scoped, generated)
        for column in ("document_name", "policy_title", "lookback_period"):
            extracted[column] = extracted[column].map(normalize_text)
        for column in ("restatement_trigger", "misconduct_trigger"):
            extracted[column] = extracted[column].map(parse_bool)
        extracted["expanded_covered_group"] = extracted[
            "expanded_covered_group"
        ].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(scoped), extracted, extraction_result, time.time() - started
        )

        qualifying = extracted.loc[
            extracted["restatement_trigger"].eq(True)
            & extracted["misconduct_trigger"].eq(True)
            & extracted["expanded_covered_group"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        projected = qualifying[
            [
                "document_name",
                "policy_title",
                "lookback_period",
                "expanded_covered_group",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(qualifying), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
