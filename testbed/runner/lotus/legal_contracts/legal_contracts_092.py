#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-092."""

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

TASK_ID = "legal_contracts-092"


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        policy_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS policies",
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
            "SEM_EXTRACT(company key and three prohibition flags)",
            input_rows=len(policies),
        ) as step:
            policy_records = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "canonical_company_key": (
                        "the company governed by the policy, as a concise canonical "
                        "company name"
                    ),
                    "prohibits_hedging": (
                        "true only if the policy prohibits hedging transactions"
                    ),
                    "prohibits_short_sales": (
                        "true only if the policy prohibits short sales"
                    ),
                    "prohibits_pledging": (
                        "true only if the policy prohibits pledging company securities"
                    ),
                },
            )
            policy_records["canonical_company_key"] = policy_records[
                "canonical_company_key"
            ].map(normalize_company_name)
            for column in (
                "prohibits_hedging",
                "prohibits_short_sales",
                "prohibits_pledging",
            ):
                policy_records[column] = policy_records[column].map(parse_bool)
            policy_records = policy_records.dropna(
                subset=["canonical_company_key"]
            )[
                [
                    "canonical_company_key",
                    "prohibits_hedging",
                    "prohibits_short_sales",
                    "prohibits_pledging",
                ]
            ].reset_index(drop=True)
            step.set_output(policy_records)

        subsidiary_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS subsidiaries",
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
            "SEM_EXTRACT(parent-company key and subsidiary count)",
            input_rows=len(subsidiary_filings),
        ) as step:
            subsidiary_records = subsidiary_filings.sem_extract(
                input_cols=["text"],
                output_cols={
                    "canonical_parent_company_key": (
                        "the explicitly identified parent company, as a concise "
                        "canonical company name"
                    ),
                    "subsidiary_count": (
                        "the number of explicitly listed subsidiary legal entities "
                        "as an integer"
                    ),
                },
            )
            subsidiary_records["canonical_parent_company_key"] = subsidiary_records[
                "canonical_parent_company_key"
            ].map(normalize_company_name)
            subsidiary_records["subsidiary_count"] = subsidiary_records[
                "subsidiary_count"
            ].map(parse_number)
            subsidiary_records = subsidiary_records.dropna(
                subset=["canonical_parent_company_key"]
            )[
                ["canonical_parent_company_key", "subsidiary_count"]
            ].reset_index(drop=True)
            step.set_output(subsidiary_records)

        joined = policy_records.merge(
            subsidiary_records,
            left_on="canonical_company_key",
            right_on="canonical_parent_company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(canonical company key = canonical parent-company key)",
            {"policies": len(policy_records), "subsidiaries": len(subsidiary_records)},
            len(joined),
            output=joined,
        )

        company_profiles = (
            joined.groupby(
                [
                    "canonical_company_key",
                    "prohibits_hedging",
                    "prohibits_short_sales",
                    "prohibits_pledging",
                ],
                sort=False,
                dropna=False,
            )["subsidiary_count"]
            .max()
            .rename("subsidiary_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(company and restriction profile, max subsidiary count)",
            len(joined),
            len(company_profiles),
            output=company_profiles,
        )

        answer_frame = (
            company_profiles.groupby(
                [
                    "prohibits_hedging",
                    "prohibits_short_sales",
                    "prohibits_pledging",
                ],
                sort=False,
                dropna=False,
            )
            .agg(
                company_count=("canonical_company_key", "size"),
                median_subsidiary_count=("subsidiary_count", "median"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(restriction profile, company count, median subsidiary count)",
            len(company_profiles),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
