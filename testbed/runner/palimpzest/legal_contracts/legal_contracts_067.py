#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-067."""

from __future__ import annotations

import os
import re
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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-067"
DATASET = "contract-exhibit"
TRIGGER_PATTERNS = ("restatement_only", "restatement_and_misconduct")
LEGAL_SUFFIXES = {
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "limited", "llc", "llp", "lp", "ltd", "plc",
}


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "not_in_scope"}
    )


def normalize_company_name(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    tokens = re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def normalize_iso_date(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


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
                "Keep the document only if it is a clawback or compensation-"
                "recovery policy."
            ),
            depends_on=["text"],
        )
        started = time.time()
        policy_result = policy_plan.run(config)
        policies = result_frame(policy_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), policies, policy_result,
            time.time() - started,
        )

        condition_plan = memory_dataset(
            f"{TASK_ID}-conditions", policies
        ).sem_filter(
            filter=(
                "Keep the policy only if it goes beyond baseline SEC clawback "
                "requirements and identifies the company whose policy it is."
            ),
            depends_on=["text"],
        )
        started = time.time()
        condition_result = condition_plan.run(config)
        qualifying = result_frame(condition_result, policies)
        tracker.record_semantic(
            "sem_filter", len(policies), qualifying, condition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {"name": "company", "type": str | None,
                 "desc": "The primary company name stated in the policy."},
                {"name": "policy_version_date", "type": str | None,
                 "desc": "The latest identifiable policy version, approval, or adoption date as YYYY-MM-DD, or null when unidentifiable."},
                {"name": "trigger_pattern", "type": str | None,
                 "desc": "Exactly restatement_only or restatement_and_misconduct."},
                {"name": "coverage_extends_beyond_executive_officers", "type": bool | None,
                 "desc": "True if covered persons extend beyond executive officers; false otherwise."},
            ],
            desc=(
                "Extract company, policy version, trigger pattern, and coverage "
                "breadth."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "company", "policy_version_date", "trigger_pattern",
            "coverage_extends_beyond_executive_officers",
        ]
        extracted = result_frame(extraction_result, qualifying, generated).rename(
            columns={"document_id": "policy_document_id"}
        )
        extracted["company"] = extracted["company"].map(normalize_text)
        extracted["company_key"] = extracted["company"].map(normalize_company_name)
        extracted["policy_version_date"] = extracted["policy_version_date"].map(
            normalize_iso_date
        )
        extracted["trigger_pattern"] = extracted["trigger_pattern"].map(
            lambda value: normalize_enum(value, TRIGGER_PATTERNS)
        )
        extracted["coverage_extends_beyond_executive_officers"] = extracted[
            "coverage_extends_beyond_executive_officers"
        ].map(parse_optional_bool)
        tracker.record_semantic(
            "sem_map", len(qualifying), extracted, extraction_result,
            time.time() - started,
        )

        latest = (
            extracted.sort_values(
                ["company_key", "policy_version_date", "policy_document_id"],
                ascending=[True, False, True],
                na_position="last",
                kind="stable",
            )
            .drop_duplicates("company_key", keep="first")
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(extracted), latest)

        projected = latest[
            ["company", "trigger_pattern", "coverage_extends_beyond_executive_officers"]
        ].reset_index(drop=True)
        tracker.record("project", len(latest), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
