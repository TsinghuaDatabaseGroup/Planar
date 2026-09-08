#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-080."""

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
    normalize_iso_date,
    parse_bool,
    parse_label_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-080"
COVERED_CATEGORIES = (
    "directors",
    "officers_or_executives",
    "employees_or_personnel",
    "external_service_providers",
    "family_or_household",
    "controlled_entities_or_accounts",
)


def main():
    setup(max_tokens=1024, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        certification_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS certification_documents",
            None,
            len(certification_documents),
            output=certification_documents,
        )

        with tracker.step(
            "SEM_FILTER(genuine EX-31 SOX Section 302 certification)",
            input_rows=len(certification_documents),
        ) as step:
            certifications = certification_documents.sem_filter(
                "The document {text} is a genuine EX-31 Sarbanes-Oxley Section 302 "
                "certification."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_EXTRACT(certification company)",
            input_rows=len(certifications),
        ) as step:
            certification_records = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "certification_company": (
                        "the company name stated in the certification; return "
                        "not_in_scope only for a non-certification"
                    )
                },
            )
            certification_records["certification_company"] = certification_records[
                "certification_company"
            ].map(clean_optional_text)
            certification_records["company_key"] = certification_records[
                "certification_company"
            ].map(normalize_company_name)
            step.set_output(certification_records)

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
        tracker.record(
            "DEDUP(normalized certification company, keep lexicographically smallest)",
            len(certification_records),
            len(certification_companies),
            output=certification_companies,
        )

        policy_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS policy_documents",
            None,
            len(policy_documents),
            output=policy_documents,
        )

        with tracker.step(
            "SEM_FILTER(genuine EX-19 insider-trading policy)",
            input_rows=len(policy_documents),
        ) as step:
            policies = policy_documents.sem_filter(
                "The document {text} is a genuine EX-19 insider-trading policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(policy controls, company, version, and covered categories)",
            input_rows=len(policies),
        ) as step:
            policy_records = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "policy_company": "the company name stated in the policy",
                    "policy_version_date": (
                        "the policy approval or adoption date formatted as YYYY-MM-DD, "
                        "or null when no such date is stated"
                    ),
                    "pre_clearance_required": (
                        "true if the policy requires trading pre-clearance"
                    ),
                    "prohibits_hedging": "true if the policy prohibits hedging",
                    "prohibits_short_selling": (
                        "true if the policy prohibits short selling"
                    ),
                    "prohibits_pledging": "true if the policy prohibits pledging",
                    "covered_person_categories": (
                        "a list of all covered categories using only directors, "
                        "officers_or_executives, employees_or_personnel, "
                        "external_service_providers, family_or_household, and "
                        "controlled_entities_or_accounts, in that order"
                    ),
                },
            )
            policy_records = policy_records.rename(
                columns={"document_id": "policy_document_id"}
            )
            policy_records["policy_company"] = policy_records["policy_company"].map(
                clean_optional_text
            )
            policy_records["company_key"] = policy_records["policy_company"].map(
                normalize_company_name
            )
            policy_records["policy_version_date"] = policy_records[
                "policy_version_date"
            ].map(normalize_iso_date)
            for column in (
                "pre_clearance_required",
                "prohibits_hedging",
                "prohibits_short_selling",
                "prohibits_pledging",
            ):
                policy_records[column] = policy_records[column].map(parse_bool)
            policy_records["covered_person_categories"] = policy_records[
                "covered_person_categories"
            ].map(lambda value: parse_label_list(value, COVERED_CATEGORIES))
            step.set_output(policy_records)

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
        tracker.record(
            "DEDUP(normalized policy company, keep latest date then document ID ASC)",
            len(policy_records),
            len(latest_policies),
            output=latest_policies,
        )

        joined = certification_companies.dropna(subset=["company_key"]).merge(
            latest_policies.dropna(subset=["company_key"]),
            on="company_key",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(certification_companies, latest_policies, on=normalized company)",
            {
                "certification_companies": len(certification_companies),
                "latest_policies": len(latest_policies),
            },
            len(joined),
            output=joined,
        )

        qualifying = joined.loc[
            joined["pre_clearance_required"]
            & joined["prohibits_hedging"]
            & joined["prohibits_short_selling"]
            & joined["prohibits_pledging"]
            & (joined["covered_person_categories"].map(len) >= 5)
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(all four policy controls AND at least five covered categories)",
            len(joined),
            len(qualifying),
            output=qualifying,
        )

        projected = qualifying[
            ["policy_company", "covered_person_categories"]
        ].rename(columns={"policy_company": "company"})
        projected = projected.reset_index(drop=True)
        tracker.record(
            "PROJECT(policy_company AS company, covered_person_categories)",
            len(qualifying),
            len(projected),
            output=projected,
        )

        ordered = projected.assign(
            category_count=projected["covered_person_categories"].map(len)
        ).sort_values(
            ["category_count", "company"],
            ascending=[False, True],
            kind="stable",
        )
        ordered = ordered.drop(columns=["category_count"]).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(CARDINALITY(categories) DESC, company ASC)",
            len(projected),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
