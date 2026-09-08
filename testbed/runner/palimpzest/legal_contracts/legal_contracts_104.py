#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-104."""

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

TASK_ID = "legal_contracts-104"
DATASET = "contract-exhibit"
LEGAL_SUFFIXES = {
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "limited", "llc", "llp", "lp", "ltd", "plc",
}


def normalize_company_name(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None or pd.isna(value):
        return None
    tokens = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        nda_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, nda_documents)

        nda_filter_plan = memory_dataset(
            f"{TASK_ID}-ndas", nda_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it has non-disclosure-agreement status, "
                "including a dual-status SEC exhibit."
            ),
            depends_on=["text"],
        )
        started = time.time()
        nda_filter_result = nda_filter_plan.run(config)
        ndas = result_frame(nda_filter_result, nda_documents)
        tracker.record_semantic(
            "sem_filter", len(nda_documents), ndas,
            nda_filter_result, time.time() - started,
        )

        restriction_filter_plan = memory_dataset(
            f"{TASK_ID}-restrictions", ndas
        ).sem_filter(
            filter=(
                "Keep the document only if it states a finite non-compete duration "
                "and contains an operative customer non-solicitation clause or an "
                "operative employee non-solicitation clause."
            ),
            depends_on=["text"],
        )
        started = time.time()
        restriction_filter_result = restriction_filter_plan.run(config)
        restrictive_ndas = result_frame(restriction_filter_result, ndas)
        tracker.record_semantic(
            "sem_filter", len(ndas), restrictive_ndas,
            restriction_filter_result, time.time() - started,
        )

        nda_extraction_plan = memory_dataset(
            f"{TASK_ID}-nda-extraction", restrictive_ndas
        ).sem_map(
            cols=[
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical company key for the stated corporate party."},
                {"name": "non_compete_duration_years", "type": float,
                 "desc": "The finite non-compete duration normalized to years."},
            ],
            desc="Extract the NDA corporate-party key and finite non-compete duration.",
            depends_on=["text"],
        )
        started = time.time()
        nda_extraction_result = nda_extraction_plan.run(config)
        nda_records = result_frame(
            nda_extraction_result,
            restrictive_ndas,
            ["company_key", "non_compete_duration_years"],
        ).rename(columns={"document_id": "nda_document_id"})
        nda_records["company_key"] = nda_records["company_key"].map(
            normalize_company_name
        )
        nda_records["non_compete_duration_years"] = nda_records[
            "non_compete_duration_years"
        ].map(normalize_number)
        nda_records = nda_records.dropna(
            subset=["company_key", "non_compete_duration_years"]
        )[["nda_document_id", "company_key", "non_compete_duration_years"]].reset_index(
            drop=True
        )
        tracker.record_semantic(
            "sem_map", len(restrictive_ndas), nda_records,
            nda_extraction_result, time.time() - started,
        )

        restrictive_profiles = nda_records[["nda_document_id", "company_key"]].copy()
        restrictive_profiles["highly_restrictive"] = nda_records[
            "non_compete_duration_years"
        ].gt(3)
        restrictive_profiles = restrictive_profiles.reset_index(drop=True)
        tracker.record("project", len(nda_records), restrictive_profiles)

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
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical company key for the company governed by the policy."},
                {"name": "has_blackout", "type": bool,
                 "desc": "True only if the policy includes a blackout period; otherwise false."},
                {"name": "has_preclearance", "type": bool,
                 "desc": "True only if the policy requires or includes pre-clearance; otherwise false."},
                {"name": "has_10b5_1", "type": bool,
                 "desc": "True only if the policy includes a Rule 10b5-1 provision; otherwise false."},
            ],
            desc="Extract the policy company key and three policy features.",
            depends_on=["text"],
        )
        started = time.time()
        policy_extraction_result = policy_extraction_plan.run(config)
        policy_records = result_frame(
            policy_extraction_result,
            policies,
            ["company_key", "has_blackout", "has_preclearance", "has_10b5_1"],
        )
        policy_records["company_key"] = policy_records["company_key"].map(
            normalize_company_name
        )
        for column in ("has_blackout", "has_preclearance", "has_10b5_1"):
            policy_records[column] = policy_records[column].map(parse_bool)
        policy_records = policy_records.dropna(subset=["company_key"])[
            ["company_key", "has_blackout", "has_preclearance", "has_10b5_1"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(policies), policy_records,
            policy_extraction_result, time.time() - started,
        )

        policy_profiles = policy_records[["company_key"]].copy()
        policy_profiles["comprehensive_policy"] = (
            policy_records["has_blackout"]
            & policy_records["has_preclearance"]
            & policy_records["has_10b5_1"]
        )
        policy_profiles = policy_profiles.reset_index(drop=True)
        tracker.record("project", len(policy_records), policy_profiles)

        unique_policy_profiles = policy_profiles.drop_duplicates(
            subset=["company_key", "comprehensive_policy"], ignore_index=True
        )
        tracker.record("dedup", len(policy_profiles), unique_policy_profiles)

        matched = restrictive_profiles.merge(
            unique_policy_profiles, on="company_key", how="inner"
        )
        tracker.record(
            "join",
            {"left": len(restrictive_profiles), "right": len(unique_policy_profiles)},
            matched,
        )

        matched["nda_company_pair"] = list(
            zip(matched["nda_document_id"], matched["company_key"])
        )
        grouped = (
            matched.groupby(
                ["highly_restrictive", "comprehensive_policy"],
                sort=False,
                dropna=False,
            )
            .agg(
                matched_nda_company_pair_count=("nda_company_pair", "nunique"),
                distinct_company_count=("company_key", "nunique"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(matched), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
