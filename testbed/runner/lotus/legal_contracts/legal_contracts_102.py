#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-102."""

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

TASK_ID = "legal_contracts-102"
EXCEPTION_COLUMNS = [
    "public_information_exception",
    "prior_knowledge_exception",
    "independent_development_exception",
    "third_party_receipt_exception",
    "legal_compulsion_exception",
    "consent_exception",
]


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        nda_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS DELAWARE_NDAS",
            None,
            len(nda_documents),
            output=nda_documents,
        )
        with tracker.step(
            "SEM_FILTER(NDA status and Delaware governing law)",
            input_rows=len(nda_documents),
        ) as step:
            delaware_ndas = nda_documents.sem_filter(
                "The document {text} has non-disclosure-agreement status, including a "
                "dual-status SEC exhibit, and expressly states Delaware as a "
                "governing-law jurisdiction."
            ).reset_index(drop=True)
            step.set_output(delaware_ndas)

        with tracker.step(
            "SEM_EXTRACT(NDA company key and six exception flags)",
            input_rows=len(delaware_ndas),
        ) as step:
            nda_records = delaware_ndas.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the stated corporate party, as a concise canonical company name"
                    ),
                    "public_information_exception": (
                        "true only if public information is expressly excluded from "
                        "confidential information"
                    ),
                    "prior_knowledge_exception": (
                        "true only if prior knowledge is expressly excluded from "
                        "confidential information"
                    ),
                    "independent_development_exception": (
                        "true only if independent development is expressly excluded from "
                        "confidential information"
                    ),
                    "third_party_receipt_exception": (
                        "true only if lawful third-party receipt is expressly excluded "
                        "from confidential information"
                    ),
                    "legal_compulsion_exception": (
                        "true only if disclosure under legal compulsion is expressly "
                        "permitted"
                    ),
                    "consent_exception": (
                        "true only if disclosure with consent is expressly permitted"
                    ),
                },
            ).rename(columns={"document_id": "nda_document_id"})
            nda_records["company_key"] = nda_records["company_key"].map(
                normalize_company_name
            )
            for column in EXCEPTION_COLUMNS:
                nda_records[column] = nda_records[column].map(parse_bool)
            nda_records = nda_records.dropna(subset=["company_key"])[
                ["nda_document_id", "company_key", *EXCEPTION_COLUMNS]
            ].reset_index(drop=True)
            step.set_output(nda_records)

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

        matched = nda_records.merge(
            certification_keys,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(DELAWARE_NDAS.company_key = CERT302.company_key)",
            {
                "DELAWARE_NDAS": len(nda_records),
                "CERT302": len(certification_keys),
            },
            len(matched),
            output=matched,
        )

        grouped = (
            matched.groupby(EXCEPTION_COLUMNS, sort=False, dropna=False)
            .agg(
                nda_document_count=("nda_document_id", "nunique"),
                distinct_matched_company_count=("company_key", "nunique"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(exception profile, NDA and company counts)",
            len(matched),
            len(grouped),
            output=grouped,
        )

        filtered = grouped[grouped["nda_document_count"] >= 2].reset_index(drop=True)
        tracker.record(
            "FILTER(nda_document_count >= 2)",
            len(grouped),
            len(filtered),
            output=filtered,
        )
        answer = df_records(filtered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
