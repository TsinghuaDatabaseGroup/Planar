#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-092."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-092"
DATASET = "contract-exhibit"
LEGAL_SUFFIXES = {
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "limited", "llc", "llp", "lp", "ltd", "plc",
}
PROFILE_COLUMNS = [
    "prohibits_hedging",
    "prohibits_short_sales",
    "prohibits_pledging",
]


def normalize_company_name(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None or pd.isna(value):
        return None
    tokens = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def normalize_integer(value) -> int | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else int(number)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        policy_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, policy_documents)

        policy_filter_plan = memory_dataset(
            f"{TASK_ID}-policies", policy_documents
        ).sem_filter(
            filter="Keep the document only if it is an EX-19 insider-trading policy.",
            depends_on=["text"],
        )
        started = time.time()
        policy_filter_result = policy_filter_plan.run(config)
        policies = result_frame(policy_filter_result, policy_documents)
        tracker.record_semantic(
            "sem_filter", len(policy_documents), policies,
            policy_filter_result, time.time() - started,
        )

        policy_extraction_plan = memory_dataset(
            f"{TASK_ID}-policy-extraction", policies
        ).sem_map(
            cols=[
                {"name": "canonical_company_key", "type": str,
                 "desc": "A concise canonical name for the company governed by the policy."},
                {"name": "prohibits_hedging", "type": bool,
                 "desc": "True only if the policy prohibits hedging transactions; otherwise false."},
                {"name": "prohibits_short_sales", "type": bool,
                 "desc": "True only if the policy prohibits short sales; otherwise false."},
                {"name": "prohibits_pledging", "type": bool,
                 "desc": "True only if the policy prohibits pledging company securities; otherwise false."},
            ],
            desc="Extract the canonical company and three restriction flags.",
            depends_on=["text"],
        )
        started = time.time()
        policy_extraction_result = policy_extraction_plan.run(config)
        policy_records = result_frame(
            policy_extraction_result,
            policies,
            ["canonical_company_key", *PROFILE_COLUMNS],
        )
        policy_records["canonical_company_key"] = policy_records[
            "canonical_company_key"
        ].map(normalize_company_name)
        for column in PROFILE_COLUMNS:
            policy_records[column] = policy_records[column].map(parse_bool)
        policy_records = policy_records.dropna(subset=["canonical_company_key"])[
            ["canonical_company_key", *PROFILE_COLUMNS]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(policies), policy_records,
            policy_extraction_result, time.time() - started,
        )

        subsidiary_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, subsidiary_documents)

        subsidiary_filter_plan = memory_dataset(
            f"{TASK_ID}-subsidiaries", subsidiary_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-21 subsidiary-list filing "
                "that explicitly identifies its parent company."
            ),
            depends_on=["text"],
        )
        started = time.time()
        subsidiary_filter_result = subsidiary_filter_plan.run(config)
        subsidiary_filings = result_frame(
            subsidiary_filter_result, subsidiary_documents
        )
        tracker.record_semantic(
            "sem_filter", len(subsidiary_documents), subsidiary_filings,
            subsidiary_filter_result, time.time() - started,
        )

        subsidiary_extraction_plan = memory_dataset(
            f"{TASK_ID}-subsidiary-extraction", subsidiary_filings
        ).sem_map(
            cols=[
                {"name": "canonical_parent_company_key", "type": str,
                 "desc": "A concise canonical name for the explicitly identified parent company."},
                {"name": "subsidiary_count", "type": int,
                 "desc": "The number of subsidiary legal entities explicitly listed in the filing."},
            ],
            desc="Extract the canonical parent-company key and subsidiary count.",
            depends_on=["text"],
        )
        started = time.time()
        subsidiary_extraction_result = subsidiary_extraction_plan.run(config)
        subsidiary_records = result_frame(
            subsidiary_extraction_result,
            subsidiary_filings,
            ["canonical_parent_company_key", "subsidiary_count"],
        )
        subsidiary_records["canonical_parent_company_key"] = subsidiary_records[
            "canonical_parent_company_key"
        ].map(normalize_company_name)
        subsidiary_records["subsidiary_count"] = subsidiary_records[
            "subsidiary_count"
        ].map(normalize_integer)
        subsidiary_records = subsidiary_records.dropna(
            subset=["canonical_parent_company_key"]
        )[["canonical_parent_company_key", "subsidiary_count"]].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(subsidiary_filings), subsidiary_records,
            subsidiary_extraction_result, time.time() - started,
        )

        joined = policy_records.merge(
            subsidiary_records,
            left_on="canonical_company_key",
            right_on="canonical_parent_company_key",
            how="inner",
        )
        tracker.record(
            "join",
            {"left": len(policy_records), "right": len(subsidiary_records)},
            joined,
        )

        company_profiles = (
            joined.groupby(
                ["canonical_company_key", *PROFILE_COLUMNS],
                sort=False,
                dropna=False,
            )["subsidiary_count"]
            .max()
            .rename("subsidiary_count")
            .reset_index()
        )
        tracker.record("groupby", len(joined), company_profiles)

        answer_frame = (
            company_profiles.groupby(PROFILE_COLUMNS, sort=False, dropna=False)
            .agg(
                company_count=("canonical_company_key", "size"),
                median_subsidiary_count=("subsidiary_count", "median"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(company_profiles), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
