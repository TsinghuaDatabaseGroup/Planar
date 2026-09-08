#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-097."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    normalize_company_name,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-097"


def _sorted_unique(values):
    return sorted({str(value) for value in values if clean_optional_text(value)})


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        certification_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS CERT302",
            None,
            len(certification_documents),
            output=certification_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-31 SOX Section 302 certification)",
            input_rows=len(certification_documents),
        ) as step:
            certifications = certification_documents.sem_filter(
                "The document {text} is an EX-31 SOX Section 302 certification."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_EXTRACT(certified company, officer, and document ID)",
            input_rows=len(certifications),
        ) as step:
            certification_records = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the certified company entity, as a concise canonical company "
                        "name"
                    ),
                    "company_name": "the company name stated in the certification",
                    "certifying_officer": (
                        "the full name of the officer who signs or certifies it"
                    ),
                },
            ).rename(columns={"document_id": "certification_document_id"})
            certification_records["company_key"] = certification_records[
                "company_key"
            ].map(normalize_company_name)
            certification_records["company_name"] = certification_records[
                "company_name"
            ].map(clean_optional_text)
            certification_records["certifying_officer"] = certification_records[
                "certifying_officer"
            ].map(clean_optional_text)
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
            step.set_output(certification_records)

        certification_groups = (
            certification_records.groupby("company_key", sort=False)
            .agg(
                company=("company_name", "min"),
                certifying_officers=("certifying_officer", _sorted_unique),
                distinct_officer_count=("certifying_officer", "nunique"),
                certification_document_count=(
                    "certification_document_id",
                    "nunique",
                ),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(company key, company, officers, certification count)",
            len(certification_records),
            len(certification_groups),
            output=certification_groups,
        )

        multi_officer_companies = certification_groups[
            certification_groups["distinct_officer_count"] >= 2
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(distinct certifying officer count >= 2)",
            len(certification_groups),
            len(multi_officer_companies),
            output=multi_officer_companies,
        )

        policy_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS POLICIES",
            None,
            len(policy_documents),
            output=policy_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-19 with none of the three prohibitions)",
            input_rows=len(policy_documents),
        ) as step:
            policies = policy_documents.sem_filter(
                "The document {text} is an EX-19 insider trading policy in which none "
                "of the prohibitions against hedging, short sales, or pledging is "
                "present."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(EX-19 canonical policy-company key)",
            input_rows=len(policies),
        ) as step:
            policy_keys = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the issuer governed by the policy, as a canonical company "
                        "name; resolve clear issuer abbreviations such as SPSL"
                    )
                },
            )
            policy_keys["company_key"] = policy_keys["company_key"].map(
                normalize_company_name
            )
            policy_keys = policy_keys.dropna(subset=["company_key"])[
                ["company_key"]
            ].reset_index(drop=True)
            step.set_output(policy_keys)

        deduplicated_policies = policy_keys.drop_duplicates(
            subset=["company_key"]
        ).reset_index(drop=True)
        tracker.record(
            "DEDUP(policy company key)",
            len(policy_keys),
            len(deduplicated_policies),
            output=deduplicated_policies,
        )

        joined = multi_officer_companies.merge(
            deduplicated_policies,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(CERT302.company_key = POLICIES.company_key)",
            {
                "CERT302": len(multi_officer_companies),
                "POLICIES": len(deduplicated_policies),
            },
            len(joined),
            output=joined,
        )

        projected = joined[
            ["company", "certifying_officers", "certification_document_count"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(company, certifying officers, certification document count)",
            len(joined),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
