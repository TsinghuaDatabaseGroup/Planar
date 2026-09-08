#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-097."""

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
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-097"
DATASET = "contract-exhibit"
LEGAL_SUFFIXES = {
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "limited", "llc", "llp", "lp", "ltd", "plc",
}


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_company_name(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    tokens = re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def sorted_unique(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value) and str(value)})


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        certification_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, certification_documents)

        certification_filter_plan = memory_dataset(
            f"{TASK_ID}-cert302", certification_documents
        ).sem_filter(
            filter="Keep the document only if it is an EX-31 SOX Section 302 certification.",
            depends_on=["text"],
        )
        started = time.time()
        certification_filter_result = certification_filter_plan.run(config)
        certifications = result_frame(
            certification_filter_result, certification_documents
        )
        tracker.record_semantic(
            "sem_filter", len(certification_documents), certifications,
            certification_filter_result, time.time() - started,
        )

        certification_extraction_plan = memory_dataset(
            f"{TASK_ID}-cert302-extraction", certifications
        ).sem_map(
            cols=[
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical name for the certified company entity."},
                {"name": "company_name", "type": str,
                 "desc": "The company name stated in the certification."},
                {"name": "certifying_officer", "type": str,
                 "desc": "The full name of the officer who signs or certifies the document."},
            ],
            desc="Extract the certified company identity and certifying officer.",
            depends_on=["text"],
        )
        started = time.time()
        certification_extraction_result = certification_extraction_plan.run(config)
        certification_records = result_frame(
            certification_extraction_result,
            certifications,
            ["company_key", "company_name", "certifying_officer"],
        ).rename(columns={"document_id": "certification_document_id"})
        certification_records["company_key"] = certification_records[
            "company_key"
        ].map(normalize_company_name)
        certification_records["company_name"] = certification_records[
            "company_name"
        ].map(normalize_text)
        certification_records["certifying_officer"] = certification_records[
            "certifying_officer"
        ].map(normalize_text)
        certification_records = certification_records.dropna(
            subset=["company_key", "company_name", "certifying_officer"]
        )[
            [
                "company_key",
                "company_name",
                "certifying_officer",
                "certification_document_id",
            ]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(certifications), certification_records,
            certification_extraction_result, time.time() - started,
        )

        certification_groups = (
            certification_records.groupby("company_key", sort=False)
            .agg(
                company=("company_name", "min"),
                certifying_officers=("certifying_officer", sorted_unique),
                distinct_officer_count=("certifying_officer", "nunique"),
                certification_document_count=("certification_document_id", "nunique"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(certification_records), certification_groups)

        multi_officer_companies = certification_groups.loc[
            certification_groups["distinct_officer_count"].ge(2)
        ].reset_index(drop=True)
        tracker.record("filter", len(certification_groups), multi_officer_companies)

        policy_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, policy_documents)

        policy_filter_plan = memory_dataset(
            f"{TASK_ID}-policies", policy_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-19 insider-trading policy in "
                "which none of the prohibitions against hedging, short sales, or "
                "pledging is present."
            ),
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
                 "desc": "A concise canonical name for the issuer governed by the policy; resolve clear issuer abbreviations such as SPSL."}
            ],
            desc="Extract the canonical policy-company key.",
            depends_on=["text"],
        )
        started = time.time()
        policy_extraction_result = policy_extraction_plan.run(config)
        policy_keys = result_frame(policy_extraction_result, policies, ["company_key"])
        policy_keys["company_key"] = policy_keys["company_key"].map(
            normalize_company_name
        )
        policy_keys = policy_keys.dropna(subset=["company_key"])[
            ["company_key"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(policies), policy_keys,
            policy_extraction_result, time.time() - started,
        )

        deduplicated_policies = policy_keys.drop_duplicates(
            subset=["company_key"], ignore_index=True
        )
        tracker.record("dedup", len(policy_keys), deduplicated_policies)

        joined = multi_officer_companies.merge(
            deduplicated_policies, on="company_key", how="inner"
        )
        tracker.record(
            "join",
            {"left": len(multi_officer_companies), "right": len(deduplicated_policies)},
            joined,
        )

        projected = joined[
            ["company", "certifying_officers", "certification_document_count"]
        ].reset_index(drop=True)
        tracker.record("project", len(joined), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
