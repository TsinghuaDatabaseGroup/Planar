#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-100."""

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

TASK_ID = "legal_contracts-100"
DATASET = "contract-exhibit"
PROFILE_COLUMNS = [
    "internal_controls_asserted",
    "fraud_disclosure_asserted",
    "material_changes_asserted",
]
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
                 "desc": "A concise canonical key for the company named in the certification."},
                {"name": "internal_controls_asserted", "type": bool,
                 "desc": "True only if the certification contains the internal-controls assertion; otherwise false."},
                {"name": "fraud_disclosure_asserted", "type": bool,
                 "desc": "True only if the certification contains the fraud-disclosure assertion; otherwise false."},
                {"name": "material_changes_asserted", "type": bool,
                 "desc": "True only if the certification contains the material-changes assertion; otherwise false."},
            ],
            desc="Extract the certified company key and three certification assertions.",
            depends_on=["text"],
        )
        started = time.time()
        certification_extraction_result = certification_extraction_plan.run(config)
        certification_records = result_frame(
            certification_extraction_result,
            certifications,
            ["company_key", *PROFILE_COLUMNS],
        ).rename(columns={"document_id": "certification_document_id"})
        certification_records["company_key"] = certification_records[
            "company_key"
        ].map(normalize_company_name)
        for column in PROFILE_COLUMNS:
            certification_records[column] = certification_records[column].map(
                parse_bool
            )
        certification_records = certification_records.dropna(subset=["company_key"])[
            ["certification_document_id", "company_key", *PROFILE_COLUMNS]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(certifications), certification_records,
            certification_extraction_result, time.time() - started,
        )

        profile_totals = (
            certification_records.groupby(PROFILE_COLUMNS, sort=False, dropna=False)
            .agg(
                certification_document_count=("certification_document_id", "nunique"),
                distinct_company_count=("company_key", "nunique"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(certification_records), profile_totals)

        clawback_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, clawback_documents)

        clawback_filter_plan = memory_dataset(
            f"{TASK_ID}-clawback", clawback_documents
        ).sem_filter(
            filter="Keep the document only if it is an EX-97 clawback policy.",
            depends_on=["text"],
        )
        started = time.time()
        clawback_filter_result = clawback_filter_plan.run(config)
        clawback_policies = result_frame(clawback_filter_result, clawback_documents)
        tracker.record_semantic(
            "sem_filter", len(clawback_documents), clawback_policies,
            clawback_filter_result, time.time() - started,
        )

        clawback_extraction_plan = memory_dataset(
            f"{TASK_ID}-clawback-extraction", clawback_policies
        ).sem_map(
            cols=[
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical key for the company governed by the policy."},
                {"name": "beyond_sec_minimum", "type": bool,
                 "desc": "True only if the policy expressly permits recovery beyond the minimum SEC financial-restatement clawback requirements; otherwise false."},
            ],
            desc="Extract the clawback company key and beyond-SEC-minimum flag.",
            depends_on=["text"],
        )
        started = time.time()
        clawback_extraction_result = clawback_extraction_plan.run(config)
        clawback_records = result_frame(
            clawback_extraction_result,
            clawback_policies,
            ["company_key", "beyond_sec_minimum"],
        )
        clawback_records["company_key"] = clawback_records["company_key"].map(
            normalize_company_name
        )
        clawback_records["beyond_sec_minimum"] = clawback_records[
            "beyond_sec_minimum"
        ].map(parse_bool)
        clawback_records = clawback_records.dropna(subset=["company_key"])[
            ["company_key", "beyond_sec_minimum"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(clawback_policies), clawback_records,
            clawback_extraction_result, time.time() - started,
        )

        matched = certification_records.merge(
            clawback_records, on="company_key", how="inner"
        )
        tracker.record(
            "join",
            {"left": len(certification_records), "right": len(clawback_records)},
            matched,
        )

        matched["beyond_minimum_company_key"] = matched["company_key"].where(
            matched["beyond_sec_minimum"]
        )
        matched_counts = (
            matched.groupby(PROFILE_COLUMNS, sort=False, dropna=False)
            .agg(
                companies_with_clawback_count=("company_key", "nunique"),
                beyond_sec_minimum_company_count=(
                    "beyond_minimum_company_key",
                    "nunique",
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(matched), matched_counts)

        joined = profile_totals.merge(
            matched_counts, on=PROFILE_COLUMNS, how="inner"
        )
        tracker.record(
            "join",
            {"left": len(profile_totals), "right": len(matched_counts)},
            joined,
        )

        projected = joined[
            [
                *PROFILE_COLUMNS,
                "certification_document_count",
                "distinct_company_count",
                "companies_with_clawback_count",
                "beyond_sec_minimum_company_count",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(joined), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
