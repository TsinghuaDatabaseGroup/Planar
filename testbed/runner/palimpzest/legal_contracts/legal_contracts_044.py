#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-044."""

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

TASK_ID = "legal_contracts-044"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        policy_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is an EX-97 compensation "
                "clawback or recoupment policy."
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

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", policies
        ).sem_filter(
            filter=(
                "Keep the policy only if it treats intentional misconduct as a "
                "recovery trigger, extends recovery beyond SEC minimum "
                "requirements, and explicitly states an effective date."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, policies)
        tracker.record_semantic(
            "sem_filter",
            len(policies),
            qualifying,
            condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {
                    "name": "document_name",
                    "type": str | None,
                    "desc": "The stated name or title of the policy.",
                },
                {
                    "name": "company_name",
                    "type": str | None,
                    "desc": "The company whose policy this is.",
                },
                {
                    "name": "effective_date",
                    "type": str | None,
                    "desc": "The explicitly stated effective date as YYYY-MM-DD.",
                },
                {
                    "name": "covered_persons",
                    "type": str | None,
                    "desc": "A concise description of the persons covered by the policy.",
                },
            ],
            desc=(
                "Extract the document name, company name, effective date, and "
                "covered-person scope."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "document_name",
            "company_name",
            "effective_date",
            "covered_persons",
        ]
        extracted = result_frame(extraction_result, qualifying, generated)
        for column in generated:
            extracted[column] = extracted[column].map(normalize_text)
        extracted = extracted[generated].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map",
            len(qualifying),
            extracted,
            extraction_result,
            time.time() - started,
        )

        answer = df_records(extracted)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
