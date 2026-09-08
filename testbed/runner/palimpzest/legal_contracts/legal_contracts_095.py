#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-095."""

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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-095"
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


def normalize_label(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or None


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
            f"{TASK_ID}-policy19", policy_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-19 insider-trading policy "
                "that expressly prohibits all three of hedging, short sales, and "
                "pledging company securities."
            ),
            depends_on=["text"],
        )
        started = time.time()
        policy_filter_result = policy_filter_plan.run(config)
        policy19 = result_frame(policy_filter_result, policy_documents)
        tracker.record_semantic(
            "sem_filter", len(policy_documents), policy19,
            policy_filter_result, time.time() - started,
        )

        policy_extraction_plan = memory_dataset(
            f"{TASK_ID}-policy19-extraction", policy19
        ).sem_map(
            cols=[
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical name for the company governed by the policy."}
            ],
            desc="Extract the canonical policy-company key.",
            depends_on=["text"],
        )
        started = time.time()
        policy_extraction_result = policy_extraction_plan.run(config)
        policy_keys = result_frame(
            policy_extraction_result, policy19, ["company_key"]
        )
        policy_keys["company_key"] = policy_keys["company_key"].map(
            normalize_company_name
        )
        policy_keys = policy_keys.dropna(subset=["company_key"])[
            ["company_key"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(policy19), policy_keys,
            policy_extraction_result, time.time() - started,
        )

        subsidiary_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, subsidiary_documents)

        subsidiary_filter_plan = memory_dataset(
            f"{TASK_ID}-subs21", subsidiary_documents
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
            f"{TASK_ID}-subs21-extraction", subsidiary_filings
        ).sem_map(
            cols=[
                {"name": "company_key", "type": str,
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

        clawback_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, clawback_documents)

        clawback_filter_plan = memory_dataset(
            f"{TASK_ID}-claw97", clawback_documents
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

        trigger_filter_plan = memory_dataset(
            f"{TASK_ID}-claw97-triggers", clawback_policies
        ).sem_filter(
            filter=(
                "Keep the clawback policy only if it provides financial-restatement "
                "recovery and also permits misconduct-based recovery independently "
                "of a financial restatement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        trigger_filter_result = trigger_filter_plan.run(config)
        qualifying_clawbacks = result_frame(
            trigger_filter_result, clawback_policies
        )
        tracker.record_semantic(
            "sem_filter", len(clawback_policies), qualifying_clawbacks,
            trigger_filter_result, time.time() - started,
        )

        clawback_extraction_plan = memory_dataset(
            f"{TASK_ID}-claw97-extraction", qualifying_clawbacks
        ).sem_map(
            cols=[
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical name for the company governed by the policy."},
                {"name": "company", "type": str,
                 "desc": "The company name stated in the policy."},
                {"name": "covered_person_category", "type": str,
                 "desc": "A concise normalized category for the people covered by the clawback policy."},
            ],
            desc="Extract the canonical key, stated company, and covered-person category.",
            depends_on=["text"],
        )
        started = time.time()
        clawback_extraction_result = clawback_extraction_plan.run(config)
        clawback_records = result_frame(
            clawback_extraction_result,
            qualifying_clawbacks,
            ["company_key", "company", "covered_person_category"],
        )
        clawback_records["company_key"] = clawback_records["company_key"].map(
            normalize_company_name
        )
        clawback_records["company"] = clawback_records["company"].map(normalize_text)
        clawback_records["covered_person_category"] = clawback_records[
            "covered_person_category"
        ].map(normalize_label)
        clawback_records = clawback_records.dropna(
            subset=["company_key", "company", "covered_person_category"]
        )[["company_key", "company", "covered_person_category"]].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(qualifying_clawbacks), clawback_records,
            clawback_extraction_result, time.time() - started,
        )

        policy_and_subsidiaries = policy_keys.merge(
            subsidiary_records, on="company_key", how="inner"
        )
        tracker.record(
            "join",
            {"left": len(policy_keys), "right": len(subsidiary_records)},
            policy_and_subsidiaries,
        )

        all_three = policy_and_subsidiaries.merge(
            clawback_records, on="company_key", how="inner"
        )
        tracker.record(
            "join",
            {"left": len(policy_and_subsidiaries), "right": len(clawback_records)},
            all_three,
        )

        grouped = (
            all_three.groupby(
                ["company", "covered_person_category"], sort=False, dropna=False
            )["subsidiary_count"]
            .max()
            .rename("subsidiary_count")
            .reset_index()
        )
        tracker.record("groupby", len(all_three), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
