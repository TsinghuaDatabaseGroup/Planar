#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-095."""

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
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-095"


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        policy19_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS POLICY19",
            None,
            len(policy19_documents),
            output=policy19_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-19 prohibits hedging, short sales, and pledging)",
            input_rows=len(policy19_documents),
        ) as step:
            policy19 = policy19_documents.sem_filter(
                "The document {text} is an EX-19 insider trading policy that expressly "
                "prohibits all three of hedging, short sales, and pledging company "
                "securities."
            ).reset_index(drop=True)
            step.set_output(policy19)

        with tracker.step(
            "SEM_EXTRACT(EX-19 canonical company key)",
            input_rows=len(policy19),
        ) as step:
            policy19_keys = policy19.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the company governed by the policy, as a concise canonical "
                        "company name"
                    )
                },
            )
            policy19_keys["company_key"] = policy19_keys["company_key"].map(
                normalize_company_name
            )
            policy19_keys = policy19_keys.dropna(subset=["company_key"])[
                ["company_key"]
            ].reset_index(drop=True)
            step.set_output(policy19_keys)

        subsidiary_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS SUBS21",
            None,
            len(subsidiary_documents),
            output=subsidiary_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-21 list with explicit parent)",
            input_rows=len(subsidiary_documents),
        ) as step:
            subsidiary_filings = subsidiary_documents.sem_filter(
                "The document {text} is an EX-21 subsidiary-list filing that "
                "explicitly identifies its parent company."
            ).reset_index(drop=True)
            step.set_output(subsidiary_filings)

        with tracker.step(
            "SEM_EXTRACT(EX-21 company key and subsidiary count)",
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
                        "the number of explicitly listed subsidiary legal entities "
                        "as an integer"
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

        clawback_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS CLAW97",
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
            "SEM_FILTER(restatement and misconduct-based recovery)",
            input_rows=len(clawback_policies),
        ) as step:
            qualifying_clawbacks = clawback_policies.sem_filter(
                "The clawback policy {text} provides financial-restatement recovery "
                "and also permits recovery based on misconduct independently of a "
                "financial restatement."
            ).reset_index(drop=True)
            step.set_output(qualifying_clawbacks)

        with tracker.step(
            "SEM_EXTRACT(EX-97 company and covered-person category)",
            input_rows=len(qualifying_clawbacks),
        ) as step:
            clawback_records = qualifying_clawbacks.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the company governed by the policy, as a concise canonical "
                        "company name"
                    ),
                    "company": "the company name stated in the policy",
                    "covered_person_category": (
                        "a concise normalized category for the people covered by the "
                        "clawback policy"
                    ),
                },
            )
            clawback_records["company_key"] = clawback_records["company_key"].map(
                normalize_company_name
            )
            clawback_records["company"] = clawback_records["company"].map(
                clean_optional_text
            )
            clawback_records["covered_person_category"] = clawback_records[
                "covered_person_category"
            ].map(clean_optional_text)
            clawback_records = clawback_records.dropna(
                subset=["company_key", "company", "covered_person_category"]
            )[
                ["company_key", "company", "covered_person_category"]
            ].reset_index(drop=True)
            step.set_output(clawback_records)

        policy_and_subsidiaries = policy19_keys.merge(
            subsidiary_records,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(POLICY19.company_key = SUBS21.company_key)",
            {"POLICY19": len(policy19_keys), "SUBS21": len(subsidiary_records)},
            len(policy_and_subsidiaries),
            output=policy_and_subsidiaries,
        )

        all_three = policy_and_subsidiaries.merge(
            clawback_records,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(POLICY19.company_key = CLAW97.company_key)",
            {
                "POLICY19_SUBS21": len(policy_and_subsidiaries),
                "CLAW97": len(clawback_records),
            },
            len(all_three),
            output=all_three,
        )

        grouped = (
            all_three.groupby(
                ["company", "covered_person_category"],
                sort=False,
                dropna=False,
            )["subsidiary_count"]
            .max()
            .rename("subsidiary_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(company, covered-person category, max subsidiary count)",
            len(all_three),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
