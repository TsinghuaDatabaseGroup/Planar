#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-084."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    normalize_free_label,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-084"


def _sorted_distinct(values):
    return sorted({str(value) for value in values if pd.notna(value) and str(value)})


def main():
    setup(max_tokens=768, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        sox_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS SOX",
            None,
            len(sox_documents),
            output=sox_documents,
        )
        with tracker.step(
            "SEM_FILTER(SOX Section 302 certification)",
            input_rows=len(sox_documents),
        ) as step:
            sox_certifications = sox_documents.sem_filter(
                "The document {text} is a genuine Sarbanes-Oxley Section 302 "
                "certification."
            ).reset_index(drop=True)
            step.set_output(sox_certifications)

        with tracker.step(
            "SEM_EXTRACT(certified company and certifying officer)",
            input_rows=len(sox_certifications),
        ) as step:
            sox_records = sox_certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "sox_company": "the certified company entity",
                    "certifying_officer": (
                        "the certifying officer's full name together with stated title"
                    ),
                },
            )
            sox_records["sox_company"] = sox_records["sox_company"].map(clean_text)
            sox_records["certifying_officer"] = sox_records[
                "certifying_officer"
            ].map(clean_text)
            step.set_output(sox_records)

        sox_groups = (
            sox_records.groupby("sox_company", sort=False)
            .agg(certifying_officers=("certifying_officer", _sorted_distinct))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(SOX company, sorted distinct certifying officers)",
            len(sox_records),
            len(sox_groups),
            output=sox_groups,
        )

        clawback_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS CLAWBACK",
            None,
            len(clawback_documents),
            output=clawback_documents,
        )
        with tracker.step(
            "SEM_FILTER(clawback or compensation-recovery policy)",
            input_rows=len(clawback_documents),
        ) as step:
            clawback_policies = clawback_documents.sem_filter(
                "The document {text} is a clawback or compensation-recovery policy."
            ).reset_index(drop=True)
            step.set_output(clawback_policies)

        with tracker.step(
            "SEM_FILTER(policy permits recovery for misconduct)",
            input_rows=len(clawback_policies),
        ) as step:
            misconduct_policies = clawback_policies.sem_filter(
                "The policy {text} permits compensation recovery for misconduct."
            ).reset_index(drop=True)
            step.set_output(misconduct_policies)

        with tracker.step(
            "SEM_EXTRACT(clawback company and covered-person category)",
            input_rows=len(misconduct_policies),
        ) as step:
            clawback_records = misconduct_policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "clawback_company": "the company entity covered by the policy",
                    "covered_person_category": (
                        "a concise normalized category for the persons covered by "
                        "the recovery policy"
                    ),
                },
            )
            clawback_records["clawback_company"] = clawback_records[
                "clawback_company"
            ].map(clean_text)
            clawback_records["covered_person_category"] = clawback_records[
                "covered_person_category"
            ].map(normalize_free_label)
            step.set_output(clawback_records)

        clawback_distinct = clawback_records[
            ["clawback_company", "covered_person_category"]
        ].drop_duplicates(ignore_index=True)
        tracker.record(
            "DEDUP(clawback company, covered-person category)",
            len(clawback_records),
            len(clawback_distinct),
            output=clawback_distinct,
        )

        subsidiary_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS SUBSIDIARIES",
            None,
            len(subsidiary_documents),
            output=subsidiary_documents,
        )
        with tracker.step(
            "SEM_FILTER(subsidiary-list filing)",
            input_rows=len(subsidiary_documents),
        ) as step:
            subsidiary_filings = subsidiary_documents.sem_filter(
                "The document {text} is a genuine subsidiary-list filing."
            ).reset_index(drop=True)
            step.set_output(subsidiary_filings)

        with tracker.step(
            "SEM_EXTRACT(parent company and subsidiary count)",
            input_rows=len(subsidiary_filings),
        ) as step:
            subsidiary_records = subsidiary_filings.sem_extract(
                input_cols=["text"],
                output_cols={
                    "subsidiary_company": (
                        "the parent company entity identified by the filing"
                    ),
                    "subsidiary_count": (
                        "the number of listed subsidiary legal entities as an integer"
                    ),
                },
            )
            subsidiary_records["subsidiary_company"] = subsidiary_records[
                "subsidiary_company"
            ].map(clean_text)
            subsidiary_records["subsidiary_count"] = subsidiary_records[
                "subsidiary_count"
            ].map(parse_number)
            step.set_output(subsidiary_records)

        large_filings = subsidiary_records.loc[
            subsidiary_records["subsidiary_count"].notna()
            & (subsidiary_records["subsidiary_count"] >= 70)
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(subsidiary_count >= 70)",
            len(subsidiary_records),
            len(large_filings),
            output=large_filings,
        )

        subsidiary_groups = (
            large_filings.groupby("subsidiary_company", sort=False)
            .agg(largest_subsidiary_count=("subsidiary_count", "max"))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(subsidiary company, MAX(subsidiary_count))",
            len(large_filings),
            len(subsidiary_groups),
            output=subsidiary_groups,
        )

        with tracker.step(
            "SEM_JOIN(SOX and clawback on same company entity)",
            input_rows={"SOX": len(sox_groups), "CLAWBACK": len(clawback_distinct)},
        ) as step:
            if sox_groups.empty or clawback_distinct.empty:
                sox_clawback = pd.DataFrame(
                    columns=[
                        "sox_company",
                        "certifying_officers",
                        "clawback_company",
                        "covered_person_category",
                    ]
                )
            else:
                sox_clawback = sox_groups.sem_join(
                    clawback_distinct,
                    "The SOX-certified company {sox_company:left} and the clawback-"
                    "policy company {clawback_company:right} are the same corporate "
                    "entity despite naming, capitalization, punctuation, or corporate-"
                    "suffix variation."
                ).reset_index(drop=True)
            step.set_output(sox_clawback)

        with tracker.step(
            "SEM_JOIN(previous matches and subsidiary filings on same company entity)",
            input_rows={
                "SOX_CLAWBACK": len(sox_clawback),
                "SUBSIDIARIES": len(subsidiary_groups),
            },
        ) as step:
            if sox_clawback.empty or subsidiary_groups.empty:
                matched = pd.DataFrame(
                    columns=[
                        "sox_company",
                        "certifying_officers",
                        "covered_person_category",
                        "largest_subsidiary_count",
                    ]
                )
            else:
                matched = sox_clawback.sem_join(
                    subsidiary_groups,
                    "The previously matched company {sox_company:left} and the parent "
                    "company {subsidiary_company:right} are the same corporate entity "
                    "despite naming, capitalization, punctuation, or corporate-suffix "
                    "variation."
                ).reset_index(drop=True)
            step.set_output(matched)

        ordered = matched.rename(columns={"sox_company": "company"})[
            [
                "company",
                "certifying_officers",
                "covered_person_category",
                "largest_subsidiary_count",
            ]
        ].sort_values(
            ["largest_subsidiary_count", "company"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(largest_subsidiary_count DESC, company ASC)",
            len(matched),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
