#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-103."""

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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-103"
DATASET = "contract-exhibit"
TRIGGER_TYPES = {"restatement_only", "restatement_and_misconduct"}
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


def normalize_trigger_type(value) -> str | None:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().casefold()
    ).strip("_")
    return normalized if normalized in TRIGGER_TYPES else None


def normalize_integer(value) -> int | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else int(number)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        clawback_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, clawback_documents)

        clawback_filter_plan = memory_dataset(
            f"{TASK_ID}-clawback", clawback_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-97 clawback policy whose "
                "recovery trigger is either financial restatement only or financial "
                "restatement plus misconduct."
            ),
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
                {"name": "trigger_type", "type": str,
                 "desc": "Exactly restatement_only when recovery is limited to financial restatement, or restatement_and_misconduct when misconduct is also an independent recovery basis."},
            ],
            desc="Extract the clawback company key and normalized trigger type.",
            depends_on=["text"],
        )
        started = time.time()
        clawback_extraction_result = clawback_extraction_plan.run(config)
        clawback_records = result_frame(
            clawback_extraction_result,
            clawback_policies,
            ["company_key", "trigger_type"],
        )
        clawback_records["company_key"] = clawback_records["company_key"].map(
            normalize_company_name
        )
        clawback_records["trigger_type"] = clawback_records["trigger_type"].map(
            normalize_trigger_type
        )
        clawback_records = clawback_records.dropna(
            subset=["company_key", "trigger_type"]
        )[["company_key", "trigger_type"]].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(clawback_policies), clawback_records,
            clawback_extraction_result, time.time() - started,
        )

        unique_clawbacks = clawback_records.drop_duplicates(
            subset=["company_key", "trigger_type"], ignore_index=True
        )
        tracker.record("dedup", len(clawback_records), unique_clawbacks)

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
                 "desc": "A concise canonical key for the company named in the certification."}
            ],
            desc="Extract the canonical certified-company key.",
            depends_on=["text"],
        )
        started = time.time()
        certification_extraction_result = certification_extraction_plan.run(config)
        certification_keys = result_frame(
            certification_extraction_result, certifications, ["company_key"]
        )
        certification_keys["company_key"] = certification_keys["company_key"].map(
            normalize_company_name
        )
        certification_keys = certification_keys.dropna(subset=["company_key"])[
            ["company_key"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(certifications), certification_keys,
            certification_extraction_result, time.time() - started,
        )

        unique_certification_keys = certification_keys.drop_duplicates(
            subset=["company_key"], ignore_index=True
        )
        tracker.record("dedup", len(certification_keys), unique_certification_keys)

        subsidiary_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, subsidiary_documents)

        subsidiary_filter_plan = memory_dataset(
            f"{TASK_ID}-subsidiaries", subsidiary_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-21 subsidiary list with an "
                "explicitly identified parent company and a stated subsidiary count."
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
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical key for the explicitly identified parent company."},
                {"name": "subsidiary_count", "type": int,
                 "desc": "The number of stated subsidiary legal entities."},
            ],
            desc="Extract the canonical parent-company key and subsidiary count.",
            depends_on=["text"],
        )
        started = time.time()
        subsidiary_extraction_result = subsidiary_extraction_plan.run(config)
        subsidiary_records = result_frame(
            subsidiary_extraction_result,
            subsidiary_filings,
            ["company_key", "subsidiary_count"],
        )
        subsidiary_records["company_key"] = subsidiary_records["company_key"].map(
            normalize_company_name
        )
        subsidiary_records["subsidiary_count"] = subsidiary_records[
            "subsidiary_count"
        ].map(normalize_integer)
        subsidiary_records = subsidiary_records.dropna(subset=["company_key"])[
            ["company_key", "subsidiary_count"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(subsidiary_filings), subsidiary_records,
            subsidiary_extraction_result, time.time() - started,
        )

        largest_counts = (
            subsidiary_records.groupby("company_key", sort=False)["subsidiary_count"]
            .max()
            .rename("largest_subsidiary_count")
            .reset_index()
        )
        tracker.record("groupby", len(subsidiary_records), largest_counts)

        clawback_with_subsidiaries = unique_clawbacks.merge(
            largest_counts, on="company_key", how="inner"
        )
        tracker.record(
            "join",
            {"left": len(unique_clawbacks), "right": len(largest_counts)},
            clawback_with_subsidiaries,
        )

        all_three = clawback_with_subsidiaries.merge(
            unique_certification_keys, on="company_key", how="inner"
        )
        tracker.record(
            "join",
            {
                "left": len(clawback_with_subsidiaries),
                "right": len(unique_certification_keys),
            },
            all_three,
        )

        grouped = (
            all_three.groupby("trigger_type", sort=False)
            .agg(
                matched_company_count=("company_key", "nunique"),
                avg_largest_subsidiary_count=("largest_subsidiary_count", "mean"),
            )
            .reset_index()
        )
        grouped["avg_largest_subsidiary_count"] = grouped[
            "avg_largest_subsidiary_count"
        ].round(2)
        tracker.record("groupby", len(all_three), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
