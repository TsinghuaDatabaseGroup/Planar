#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-088."""

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
    save_output,
    setup,
)

TASK_ID = "legal_contracts-088"


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
            "SEM_FILTER(EX-31 SOX officer certification)",
            input_rows=len(sox_documents),
        ) as step:
            certifications = sox_documents.sem_filter(
                "The document {text} is a genuine EX-31 SOX officer certification."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_EXTRACT(certified company and officer with title)",
            input_rows=len(certifications),
        ) as step:
            sox_records = certifications.sem_extract(
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
            "SEM_FILTER(EX-97 clawback policy)",
            input_rows=len(clawback_documents),
        ) as step:
            clawback_policies = clawback_documents.sem_filter(
                "The document {text} is a genuine EX-97 clawback policy."
            ).reset_index(drop=True)
            step.set_output(clawback_policies)

        with tracker.step(
            "SEM_FILTER(restatement recovery and misconduct recovery)",
            input_rows=len(clawback_policies),
        ) as step:
            dual_trigger = clawback_policies.sem_filter(
                "The policy {text} provides recovery for a financial restatement and "
                "also permits recovery for misconduct. Both triggers must be present."
            ).reset_index(drop=True)
            step.set_output(dual_trigger)

        with tracker.step(
            "SEM_EXTRACT(clawback company and covered-person category)",
            input_rows=len(dual_trigger),
        ) as step:
            clawback_records = dual_trigger.sem_extract(
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

        with tracker.step(
            "SEM_JOIN(SOX and clawback on same company entity)",
            input_rows={"SOX": len(sox_groups), "CLAWBACK": len(clawback_distinct)},
        ) as step:
            if sox_groups.empty or clawback_distinct.empty:
                matched = pd.DataFrame(
                    columns=[
                        "sox_company",
                        "certifying_officers",
                        "covered_person_category",
                    ]
                )
            else:
                matched = sox_groups.sem_join(
                    clawback_distinct,
                    "The SOX-certified company {sox_company:left} and the clawback-"
                    "policy company {clawback_company:right} are the same corporate "
                    "entity despite naming, capitalization, punctuation, or corporate-"
                    "suffix variation."
                ).reset_index(drop=True)
            step.set_output(matched)

        projected = matched.rename(columns={"sox_company": "company"})[
            ["company", "certifying_officers", "covered_person_category"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(company, certifying officers, covered-person category)",
            len(matched),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
