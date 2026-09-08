#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-080."""

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
    parse_bool,
    parse_label_list,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-080"
DATASET = "contract-exhibit"
COVERED_CATEGORIES = (
    "directors",
    "officers_or_executives",
    "employees_or_personnel",
    "external_service_providers",
    "family_or_household",
    "controlled_entities_or_accounts",
)
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


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        certification_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, certification_documents)

        certification_plan = memory_dataset(
            f"{TASK_ID}-certification-filter", certification_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is a genuine EX-31 "
                "Sarbanes-Oxley Section 302 certification."
            ),
            depends_on=["text"],
        )
        started = time.time()
        certification_result = certification_plan.run(config)
        certifications = result_frame(
            certification_result, certification_documents
        )
        tracker.record_semantic(
            "sem_filter", len(certification_documents), certifications,
            certification_result, time.time() - started,
        )

        certification_extraction_plan = memory_dataset(
            f"{TASK_ID}-certification-extraction", certifications
        ).sem_map(
            cols=[
                {
                    "name": "certification_company",
                    "type": str | None,
                    "desc": (
                        "The company name stated in the certification; use "
                        "not_in_scope only for a non-certification."
                    ),
                }
            ],
            desc="Extract the stated certification company.",
            depends_on=["text"],
        )
        started = time.time()
        certification_extraction_result = certification_extraction_plan.run(config)
        certification_records = result_frame(
            certification_extraction_result,
            certifications,
            ["certification_company"],
        )
        certification_records["certification_company"] = certification_records[
            "certification_company"
        ].map(normalize_text)
        certification_records["company_key"] = certification_records[
            "certification_company"
        ].map(normalize_company_name)
        tracker.record_semantic(
            "sem_map", len(certifications), certification_records,
            certification_extraction_result, time.time() - started,
        )

        certification_companies = (
            certification_records.sort_values(
                ["company_key", "certification_company"],
                ascending=[True, True],
                na_position="last",
                kind="stable",
            )
            .drop_duplicates("company_key", keep="first")
            [["company_key", "certification_company"]]
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(certification_records), certification_companies)

        policy_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, policy_documents)

        policy_plan = memory_dataset(
            f"{TASK_ID}-policy-filter", policy_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is a genuine EX-19 "
                "insider-trading policy."
            ),
            depends_on=["text"],
        )
        started = time.time()
        policy_result = policy_plan.run(config)
        policies = result_frame(policy_result, policy_documents)
        tracker.record_semantic(
            "sem_filter", len(policy_documents), policies, policy_result,
            time.time() - started,
        )

        policy_extraction_plan = memory_dataset(
            f"{TASK_ID}-policy-extraction", policies
        ).sem_map(
            cols=[
                {
                    "name": "policy_company",
                    "type": str | None,
                    "desc": "The company name stated in the policy.",
                },
                {
                    "name": "policy_version_date",
                    "type": str | None,
                    "desc": (
                        "The policy approval or adoption date as YYYY-MM-DD, or "
                        "null when no such date is stated."
                    ),
                },
                {
                    "name": "pre_clearance_required",
                    "type": bool,
                    "desc": "True if the policy requires trading pre-clearance.",
                },
                {
                    "name": "prohibits_hedging",
                    "type": bool,
                    "desc": "True if the policy prohibits hedging.",
                },
                {
                    "name": "prohibits_short_selling",
                    "type": bool,
                    "desc": "True if the policy prohibits short selling.",
                },
                {
                    "name": "prohibits_pledging",
                    "type": bool,
                    "desc": "True if the policy prohibits pledging.",
                },
                {
                    "name": "covered_person_categories",
                    "type": list[str],
                    "desc": (
                        "All covered categories, in this canonical order and "
                        "using only: directors, officers_or_executives, "
                        "employees_or_personnel, external_service_providers, "
                        "family_or_household, controlled_entities_or_accounts."
                    ),
                },
            ],
            desc=(
                "Extract policy company, version date, four policy controls, "
                "and canonical covered-person categories."
            ),
            depends_on=["text"],
        )
        started = time.time()
        policy_extraction_result = policy_extraction_plan.run(config)
        policy_fields = [
            "policy_company", "policy_version_date", "pre_clearance_required",
            "prohibits_hedging", "prohibits_short_selling", "prohibits_pledging",
            "covered_person_categories",
        ]
        policy_records = result_frame(
            policy_extraction_result, policies, policy_fields
        ).rename(columns={"document_id": "policy_document_id"})
        policy_records["policy_company"] = policy_records["policy_company"].map(
            normalize_text
        )
        policy_records["company_key"] = policy_records["policy_company"].map(
            normalize_company_name
        )
        policy_records["policy_version_date"] = policy_records[
            "policy_version_date"
        ].map(normalize_iso_date)
        for column in (
            "pre_clearance_required", "prohibits_hedging",
            "prohibits_short_selling", "prohibits_pledging",
        ):
            policy_records[column] = policy_records[column].map(parse_bool)
        policy_records["covered_person_categories"] = policy_records[
            "covered_person_categories"
        ].map(lambda value: parse_label_list(value, COVERED_CATEGORIES))
        tracker.record_semantic(
            "sem_map", len(policies), policy_records, policy_extraction_result,
            time.time() - started,
        )

        latest_policies = (
            policy_records.sort_values(
                ["company_key", "policy_version_date", "policy_document_id"],
                ascending=[True, False, True],
                na_position="last",
                kind="stable",
            )
            .drop_duplicates("company_key", keep="first")
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(policy_records), latest_policies)

        joined = certification_companies.dropna(subset=["company_key"]).merge(
            latest_policies.dropna(subset=["company_key"]),
            on="company_key",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "certification_companies": len(certification_companies),
                "latest_policies": len(latest_policies),
            },
            joined,
        )

        qualifying = joined.loc[
            joined["pre_clearance_required"]
            & joined["prohibits_hedging"]
            & joined["prohibits_short_selling"]
            & joined["prohibits_pledging"]
            & (joined["covered_person_categories"].map(len) >= 5)
        ].reset_index(drop=True)
        tracker.record("filter", len(joined), qualifying)

        projected = qualifying[
            ["policy_company", "covered_person_categories"]
        ].rename(columns={"policy_company": "company"}).reset_index(drop=True)
        tracker.record("project", len(qualifying), projected)

        ordered = projected.assign(
            category_count=projected["covered_person_categories"].map(len)
        ).sort_values(
            ["category_count", "company"],
            ascending=[False, True],
            kind="stable",
        ).drop(columns=["category_count"]).reset_index(drop=True)
        tracker.record("orderby", len(projected), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
