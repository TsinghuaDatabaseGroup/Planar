#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-100."""

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
    save_output,
    setup,
)

TASK_ID = "legal_contracts-100"
PROFILE_COLUMNS = [
    "internal_controls_asserted",
    "fraud_disclosure_asserted",
    "material_changes_asserted",
]


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
            "SEM_EXTRACT(company key and three certification assertions)",
            input_rows=len(certifications),
        ) as step:
            certification_records = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the company named in the certification, as a concise canonical "
                        "company name"
                    ),
                    "internal_controls_asserted": (
                        "true only if the certification contains the internal-controls "
                        "assertion"
                    ),
                    "fraud_disclosure_asserted": (
                        "true only if the certification contains the fraud-disclosure "
                        "assertion"
                    ),
                    "material_changes_asserted": (
                        "true only if the certification contains the material-changes "
                        "assertion"
                    ),
                },
            ).rename(columns={"document_id": "certification_document_id"})
            certification_records["company_key"] = certification_records[
                "company_key"
            ].map(normalize_company_name)
            for column in PROFILE_COLUMNS:
                certification_records[column] = certification_records[column].map(
                    parse_bool
                )
            certification_records = certification_records.dropna(
                subset=["company_key"]
            )[
                ["certification_document_id", "company_key", *PROFILE_COLUMNS]
            ].reset_index(drop=True)
            step.set_output(certification_records)

        profile_totals = (
            certification_records.groupby(
                PROFILE_COLUMNS,
                sort=False,
                dropna=False,
            )
            .agg(
                certification_document_count=(
                    "certification_document_id",
                    "nunique",
                ),
                distinct_company_count=("company_key", "nunique"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(assertion profile, certification and company counts)",
            len(certification_records),
            len(profile_totals),
            output=profile_totals,
        )

        clawback_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS CLAWBACK",
            None,
            len(clawback_documents),
            output=clawback_documents,
        )
        with tracker.step(
            "SEM_FILTER(document is an EX-97 clawback policy)",
            input_rows=len(clawback_documents),
        ) as step:
            clawback_policies = clawback_documents.sem_filter(
                "The document {text} is an EX-97 clawback policy."
            ).reset_index(drop=True)
            step.set_output(clawback_policies)

        with tracker.step(
            "SEM_EXTRACT(clawback company key and beyond-minimum flag)",
            input_rows=len(clawback_policies),
        ) as step:
            clawback_records = clawback_policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the company governed by the policy, as a concise canonical "
                        "company name"
                    ),
                    "beyond_sec_minimum": (
                        "true only if the policy expressly permits recovery beyond the "
                        "minimum SEC financial-restatement clawback requirements"
                    ),
                },
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
            step.set_output(clawback_records)

        matched = certification_records.merge(
            clawback_records,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(CERT302.company_key = CLAWBACK.company_key)",
            {
                "CERT302": len(certification_records),
                "CLAWBACK": len(clawback_records),
            },
            len(matched),
            output=matched,
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
        tracker.record(
            "GROUP_BY(assertion profile, clawback company counts)",
            len(matched),
            len(matched_counts),
            output=matched_counts,
        )

        joined = profile_totals.merge(
            matched_counts,
            on=PROFILE_COLUMNS,
            how="inner",
        )
        tracker.record(
            "JOIN(PROFILE_TOTALS and MATCHED on assertion profile)",
            {
                "PROFILE_TOTALS": len(profile_totals),
                "MATCHED": len(matched_counts),
            },
            len(joined),
            output=joined,
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
        tracker.record(
            "PROJECT(profile and four counts)",
            len(joined),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
