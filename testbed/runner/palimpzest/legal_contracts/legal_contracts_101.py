#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-101."""

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

TASK_ID = "legal_contracts-101"
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


def normalize_person_key(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()) or None


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        employment_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, employment_documents)

        employment_filter_plan = memory_dataset(
            f"{TASK_ID}-employment", employment_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-10 employment agreement with "
                "a finite, expressly stated employee non-solicitation period."
            ),
            depends_on=["text"],
        )
        started = time.time()
        employment_filter_result = employment_filter_plan.run(config)
        employment = result_frame(employment_filter_result, employment_documents)
        tracker.record_semantic(
            "sem_filter", len(employment_documents), employment,
            employment_filter_result, time.time() - started,
        )

        employment_extraction_plan = memory_dataset(
            f"{TASK_ID}-employment-extraction", employment
        ).sem_map(
            cols=[
                {"name": "company_name", "type": str,
                 "desc": "The employer name stated in the agreement."},
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical company key for the employer."},
                {"name": "employee_name", "type": str,
                 "desc": "The full name of the employee."},
                {"name": "person_key", "type": str,
                 "desc": "A normalized person key resolving clear middle-name, initial, nickname, and shortened-name variants."},
                {"name": "non_solicitation_duration_years", "type": float,
                 "desc": "The finite employee non-solicitation period normalized to years."},
            ],
            desc="Extract the employment document's company, employee, keys, and duration.",
            depends_on=["text"],
        )
        started = time.time()
        employment_extraction_result = employment_extraction_plan.run(config)
        employment_records = result_frame(
            employment_extraction_result,
            employment,
            [
                "company_name",
                "company_key",
                "employee_name",
                "person_key",
                "non_solicitation_duration_years",
            ],
        ).rename(columns={"document_id": "employment_document"})
        employment_records["company_name"] = employment_records["company_name"].map(
            normalize_text
        )
        employment_records["company_key"] = employment_records["company_key"].map(
            normalize_company_name
        )
        employment_records["employee_name"] = employment_records[
            "employee_name"
        ].map(normalize_text)
        employment_records["person_key"] = employment_records["person_key"].map(
            normalize_person_key
        )
        employment_records["non_solicitation_duration_years"] = employment_records[
            "non_solicitation_duration_years"
        ].map(normalize_number)
        employment_records = employment_records.dropna(
            subset=[
                "company_name",
                "company_key",
                "employee_name",
                "person_key",
                "non_solicitation_duration_years",
            ]
        )[
            [
                "employment_document",
                "company_name",
                "company_key",
                "employee_name",
                "person_key",
                "non_solicitation_duration_years",
            ]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(employment), employment_records,
            employment_extraction_result, time.time() - started,
        )

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
                 "desc": "A concise canonical company key for the certified company."},
                {"name": "officer_name", "type": str,
                 "desc": "The full name of the certifying officer."},
                {"name": "person_key", "type": str,
                 "desc": "A normalized person key resolving clear middle-name, initial, nickname, and shortened-name variants."},
                {"name": "officer_title", "type": str,
                 "desc": "The officer title stated in the certification."},
            ],
            desc="Extract the certified company, officer, normalized person key, and title.",
            depends_on=["text"],
        )
        started = time.time()
        certification_extraction_result = certification_extraction_plan.run(config)
        certification_records = result_frame(
            certification_extraction_result,
            certifications,
            ["company_key", "officer_name", "person_key", "officer_title"],
        )
        certification_records["company_key"] = certification_records[
            "company_key"
        ].map(normalize_company_name)
        certification_records["officer_name"] = certification_records[
            "officer_name"
        ].map(normalize_text)
        certification_records["person_key"] = certification_records[
            "person_key"
        ].map(normalize_person_key)
        certification_records["officer_title"] = certification_records[
            "officer_title"
        ].map(normalize_text)
        certification_records = certification_records.dropna(
            subset=["company_key", "officer_name", "person_key", "officer_title"]
        )[["company_key", "officer_name", "person_key", "officer_title"]].reset_index(
            drop=True
        )
        tracker.record_semantic(
            "sem_map", len(certifications), certification_records,
            certification_extraction_result, time.time() - started,
        )

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
                {"name": "covered_person_scope", "type": str,
                 "desc": "The policy's stated categories, roles, or titles of covered people."},
            ],
            desc="Extract the policy company key and covered-person scope.",
            depends_on=["text"],
        )
        started = time.time()
        policy_extraction_result = policy_extraction_plan.run(config)
        policy_records = result_frame(
            policy_extraction_result,
            policies,
            ["company_key", "covered_person_scope"],
        )
        policy_records["company_key"] = policy_records["company_key"].map(
            normalize_company_name
        )
        policy_records["covered_person_scope"] = policy_records[
            "covered_person_scope"
        ].map(normalize_text)
        policy_records = policy_records.dropna(
            subset=["company_key", "covered_person_scope"]
        )[["company_key", "covered_person_scope"]].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(policies), policy_records,
            policy_extraction_result, time.time() - started,
        )

        employee_officers = employment_records.merge(
            certification_records, on=["company_key", "person_key"], how="inner"
        )
        tracker.record(
            "join",
            {"left": len(employment_records), "right": len(certification_records)},
            employee_officers,
        )

        matched = employee_officers.merge(
            policy_records, on="company_key", how="inner"
        )
        tracker.record(
            "join",
            {"left": len(employee_officers), "right": len(policy_records)},
            matched,
        )

        if matched.empty:
            covered = matched.copy()
            tracker.record("sem_filter", len(matched), covered)
        else:
            coverage_plan = memory_dataset(
                f"{TASK_ID}-coverage", matched
            ).sem_filter(
                filter=(
                    "Keep the record only if the certifying officer's title falls "
                    "within the insider-trading policy's stated covered-person scope."
                ),
                depends_on=["officer_title", "covered_person_scope"],
            )
            started = time.time()
            coverage_result = coverage_plan.run(config)
            covered = result_frame(coverage_result, matched)
            tracker.record_semantic(
                "sem_filter", len(matched), covered,
                coverage_result, time.time() - started,
            )

        projected = covered[
            [
                "company_name",
                "officer_name",
                "employment_document",
                "non_solicitation_duration_years",
            ]
        ].rename(columns={"company_name": "company", "officer_name": "individual"})
        projected = projected.reset_index(drop=True)
        tracker.record("project", len(covered), projected)

        deduplicated = projected.drop_duplicates(
            subset=[
                "company",
                "individual",
                "employment_document",
                "non_solicitation_duration_years",
            ],
            ignore_index=True,
        )
        tracker.record("dedup", len(projected), deduplicated)
        answer = df_records(deduplicated)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
