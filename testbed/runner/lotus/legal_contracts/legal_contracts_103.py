#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-103."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_company_name,
    normalize_enum,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-103"
TRIGGER_TYPES = ("restatement_only", "restatement_and_misconduct")


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        clawback_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS CLAWBACK",
            None,
            len(clawback_documents),
            output=clawback_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-97 with one of the two trigger types)",
            input_rows=len(clawback_documents),
        ) as step:
            clawback_policies = clawback_documents.sem_filter(
                "The document {text} is an EX-97 clawback policy whose recovery "
                "trigger is either financial restatement only or financial restatement "
                "plus misconduct."
            ).reset_index(drop=True)
            step.set_output(clawback_policies)

        with tracker.step(
            "SEM_EXTRACT(clawback company key and trigger type)",
            input_rows=len(clawback_policies),
        ) as step:
            clawback_records = clawback_policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the company governed by the policy, as a concise canonical "
                        "company name"
                    ),
                    "trigger_type": (
                        "exactly restatement_only when recovery is limited to financial "
                        "restatement, or restatement_and_misconduct when misconduct is "
                        "also an independent recovery basis"
                    ),
                },
            )
            clawback_records["company_key"] = clawback_records["company_key"].map(
                normalize_company_name
            )
            clawback_records["trigger_type"] = clawback_records["trigger_type"].map(
                lambda value: normalize_enum(value, TRIGGER_TYPES)
            )
            clawback_records = clawback_records.dropna(
                subset=["company_key", "trigger_type"]
            )[
                ["company_key", "trigger_type"]
            ].reset_index(drop=True)
            step.set_output(clawback_records)

        unique_clawbacks = clawback_records.drop_duplicates(
            subset=["company_key", "trigger_type"]
        ).reset_index(drop=True)
        tracker.record(
            "DEDUP(company key, trigger type)",
            len(clawback_records),
            len(unique_clawbacks),
            output=unique_clawbacks,
        )

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
            "SEM_EXTRACT(certification company key)",
            input_rows=len(certifications),
        ) as step:
            certification_keys = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the company named in the certification, as a concise canonical "
                        "company name"
                    )
                },
            )
            certification_keys["company_key"] = certification_keys[
                "company_key"
            ].map(normalize_company_name)
            certification_keys = certification_keys.dropna(subset=["company_key"])[
                ["company_key"]
            ].reset_index(drop=True)
            step.set_output(certification_keys)

        unique_certification_keys = certification_keys.drop_duplicates(
            subset=["company_key"]
        ).reset_index(drop=True)
        tracker.record(
            "DEDUP(CERT302 company key)",
            len(certification_keys),
            len(unique_certification_keys),
            output=unique_certification_keys,
        )

        subsidiary_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS SUBSIDIARIES",
            None,
            len(subsidiary_documents),
            output=subsidiary_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-21 with explicit parent and subsidiary count)",
            input_rows=len(subsidiary_documents),
        ) as step:
            subsidiary_filings = subsidiary_documents.sem_filter(
                "The document {text} is an EX-21 subsidiary list with an explicitly "
                "identified parent company and a stated subsidiary count."
            ).reset_index(drop=True)
            step.set_output(subsidiary_filings)

        with tracker.step(
            "SEM_EXTRACT(parent-company key and subsidiary count)",
            input_rows=len(subsidiary_filings),
        ) as step:
            subsidiary_records = subsidiary_filings.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the explicitly identified parent company, as a concise "
                        "canonical company name"
                    ),
                    "subsidiary_count": (
                        "the number of stated subsidiary legal entities as an integer"
                    ),
                },
            )
            subsidiary_records["company_key"] = subsidiary_records[
                "company_key"
            ].map(normalize_company_name)
            subsidiary_records["subsidiary_count"] = subsidiary_records[
                "subsidiary_count"
            ].map(parse_number)
            subsidiary_records = subsidiary_records.dropna(subset=["company_key"])[
                ["company_key", "subsidiary_count"]
            ].reset_index(drop=True)
            step.set_output(subsidiary_records)

        largest_subsidiary_counts = (
            subsidiary_records.groupby("company_key", sort=False)["subsidiary_count"]
            .max()
            .rename("largest_subsidiary_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(company key, largest subsidiary count)",
            len(subsidiary_records),
            len(largest_subsidiary_counts),
            output=largest_subsidiary_counts,
        )

        clawback_with_subsidiaries = unique_clawbacks.merge(
            largest_subsidiary_counts,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(CLAWBACK.company_key = SUBSIDIARIES.company_key)",
            {
                "CLAWBACK": len(unique_clawbacks),
                "SUBSIDIARIES": len(largest_subsidiary_counts),
            },
            len(clawback_with_subsidiaries),
            output=clawback_with_subsidiaries,
        )

        all_three = clawback_with_subsidiaries.merge(
            unique_certification_keys,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(CLAWBACK.company_key = CERT302.company_key)",
            {
                "CLAWBACK_SUBSIDIARIES": len(clawback_with_subsidiaries),
                "CERT302": len(unique_certification_keys),
            },
            len(all_three),
            output=all_three,
        )

        grouped = (
            all_three.groupby("trigger_type", sort=False)
            .agg(
                matched_company_count=("company_key", "nunique"),
                avg_largest_subsidiary_count=(
                    "largest_subsidiary_count",
                    "mean",
                ),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_largest_subsidiary_count"] = grouped[
                "avg_largest_subsidiary_count"
            ].round(2)
        tracker.record(
            "GROUP_BY(trigger type, company count, average largest subsidiary count)",
            len(all_three),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
