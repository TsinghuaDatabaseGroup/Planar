#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-104."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_company_name,
    parse_bool,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-104"


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        nda_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS RESTRICTIVE_NDAS",
            None,
            len(nda_documents),
            output=nda_documents,
        )
        with tracker.step(
            "SEM_FILTER(document has NDA status)",
            input_rows=len(nda_documents),
        ) as step:
            ndas = nda_documents.sem_filter(
                "The document {text} has non-disclosure-agreement status, including a "
                "dual-status SEC exhibit."
            ).reset_index(drop=True)
            step.set_output(ndas)

        with tracker.step(
            "SEM_FILTER(finite non-compete and customer or employee non-solicitation)",
            input_rows=len(ndas),
        ) as step:
            restrictive_ndas = ndas.sem_filter(
                "The document {text} states a finite non-compete duration and contains "
                "an operative customer non-solicitation clause or an operative employee "
                "non-solicitation clause."
            ).reset_index(drop=True)
            step.set_output(restrictive_ndas)

        with tracker.step(
            "SEM_EXTRACT(NDA company key and non-compete duration)",
            input_rows=len(restrictive_ndas),
        ) as step:
            nda_records = restrictive_ndas.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the stated corporate party, as a concise canonical company name"
                    ),
                    "non_compete_duration_years": (
                        "the finite non-compete duration converted to years"
                    ),
                },
            ).rename(columns={"document_id": "nda_document_id"})
            nda_records["company_key"] = nda_records["company_key"].map(
                normalize_company_name
            )
            nda_records["non_compete_duration_years"] = nda_records[
                "non_compete_duration_years"
            ].map(parse_number)
            nda_records = nda_records.dropna(
                subset=["company_key", "non_compete_duration_years"]
            )[
                ["nda_document_id", "company_key", "non_compete_duration_years"]
            ].reset_index(drop=True)
            step.set_output(nda_records)

        restrictive_profiles = nda_records[
            ["nda_document_id", "company_key"]
        ].copy()
        restrictive_profiles["highly_restrictive"] = (
            nda_records["non_compete_duration_years"] > 3
        )
        tracker.record(
            "PROJECT(NDA document, company key, duration > 3)",
            len(nda_records),
            len(restrictive_profiles),
            output=restrictive_profiles,
        )

        policy_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS POLICIES",
            None,
            len(policy_documents),
            output=policy_documents,
        )
        with tracker.step(
            "SEM_FILTER(document is an EX-19 insider trading policy)",
            input_rows=len(policy_documents),
        ) as step:
            policies = policy_documents.sem_filter(
                "The document {text} is an EX-19 insider trading policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(policy company key and three policy features)",
            input_rows=len(policies),
        ) as step:
            policy_records = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the company governed by the policy, as a concise canonical "
                        "company name"
                    ),
                    "has_blackout": (
                        "true only if the policy includes a blackout period"
                    ),
                    "has_preclearance": (
                        "true only if the policy requires or includes pre-clearance"
                    ),
                    "has_10b5_1": (
                        "true only if the policy includes a Rule 10b5-1 provision"
                    ),
                },
            )
            policy_records["company_key"] = policy_records["company_key"].map(
                normalize_company_name
            )
            for column in ("has_blackout", "has_preclearance", "has_10b5_1"):
                policy_records[column] = policy_records[column].map(parse_bool)
            policy_records = policy_records.dropna(subset=["company_key"])[
                ["company_key", "has_blackout", "has_preclearance", "has_10b5_1"]
            ].reset_index(drop=True)
            step.set_output(policy_records)

        policy_profiles = policy_records[["company_key"]].copy()
        policy_profiles["comprehensive_policy"] = (
            policy_records["has_blackout"]
            & policy_records["has_preclearance"]
            & policy_records["has_10b5_1"]
        )
        tracker.record(
            "PROJECT(company key, all three policy features)",
            len(policy_records),
            len(policy_profiles),
            output=policy_profiles,
        )

        unique_policy_profiles = policy_profiles.drop_duplicates(
            subset=["company_key", "comprehensive_policy"]
        ).reset_index(drop=True)
        tracker.record(
            "DEDUP(company key, comprehensive policy)",
            len(policy_profiles),
            len(unique_policy_profiles),
            output=unique_policy_profiles,
        )

        matched = restrictive_profiles.merge(
            unique_policy_profiles,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(RESTRICTIVE_NDAS.company_key = POLICIES.company_key)",
            {
                "RESTRICTIVE_NDAS": len(restrictive_profiles),
                "POLICIES": len(unique_policy_profiles),
            },
            len(matched),
            output=matched,
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
        tracker.record(
            "GROUP_BY(restriction and policy profile, pair and company counts)",
            len(matched),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
