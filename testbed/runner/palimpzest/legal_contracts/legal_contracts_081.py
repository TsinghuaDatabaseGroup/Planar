#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-081."""

from __future__ import annotations

import os
import re
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

TASK_ID = "legal_contracts-081"
DATASET = "contract-exhibit"
LEGAL_SUFFIXES = {
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "limited", "llc", "llp", "lp", "ltd", "plc",
}


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_company_name(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    tokens = re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        policy_plan = memory_dataset(
            f"{TASK_ID}-policies", documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is a genuine EX-97 clawback or "
                "compensation-recovery policy."
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

        enhanced_plan = memory_dataset(
            f"{TASK_ID}-enhanced", policies
        ).sem_filter(
            filter=(
                "Keep the policy only if it goes beyond baseline SEC clawback "
                "requirements and includes misconduct as a recovery trigger."
            ),
            depends_on=["text"],
        )
        started = time.time()
        enhanced_result = enhanced_plan.run(config)
        enhanced = result_frame(enhanced_result, policies)
        tracker.record_semantic(
            "sem_filter",
            len(policies),
            enhanced,
            enhanced_result,
            time.time() - started,
        )

        coverage_plan = memory_dataset(
            f"{TASK_ID}-coverage", enhanced
        ).sem_filter(
            filter=(
                "Keep the policy only if it explicitly extends recovery beyond "
                "executive officers to at least one of non-executive employees, "
                "directors, or key managers."
            ),
            depends_on=["text"],
        )
        started = time.time()
        coverage_result = coverage_plan.run(config)
        broader_coverage = result_frame(coverage_result, enhanced)
        tracker.record_semantic(
            "sem_filter",
            len(enhanced),
            broader_coverage,
            coverage_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", broader_coverage
        ).sem_map(
            cols=[
                {"name": "company", "type": str | None,
                 "desc": "The primary company name stated in the policy."}
            ],
            desc="Extract the primary stated company name.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result, broader_coverage, ["company"])
        extracted["company"] = extracted["company"].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(broader_coverage), extracted, extraction_result,
            time.time() - started,
        )

        deduplicated = extracted[["company"]].copy()
        deduplicated["company_key"] = deduplicated["company"].map(
            normalize_company_name
        )
        deduplicated = deduplicated.drop_duplicates(
            "company_key", keep="first"
        )[["company"]].reset_index(drop=True)
        tracker.record("dedup", len(extracted), deduplicated)
        answer = df_records(deduplicated)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
