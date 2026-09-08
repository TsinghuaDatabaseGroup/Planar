#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-099."""

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
    normalize_enum,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-099"
NDA_CATEGORIES = (
    "mutual_nda",
    "unilateral_nda",
    "confidentiality_standstill",
)


def _sorted_unique(values):
    return sorted({str(value) for value in values if clean_optional_text(value)})


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        nda_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS NDAS",
            None,
            len(nda_documents),
            output=nda_documents,
        )
        with tracker.step(
            "SEM_FILTER(one of the three specified NDA categories)",
            input_rows=len(nda_documents),
        ) as step:
            ndas = nda_documents.sem_filter(
                "The document {text} is a mutual NDA, unilateral NDA, or "
                "confidentiality-and-standstill agreement, including a dual-status "
                "SEC exhibit."
            ).reset_index(drop=True)
            step.set_output(ndas)

        with tracker.step(
            "SEM_EXTRACT(NDA category, corporate party, and canonical key)",
            input_rows=len(ndas),
        ) as step:
            nda_records = ndas.sem_extract(
                input_cols=["text"],
                output_cols={
                    "nda_category": (
                        "exactly mutual_nda, unilateral_nda, or "
                        "confidentiality_standstill"
                    ),
                    "party_name": "the stated corporate party to the agreement",
                    "party_key": (
                        "that corporate party as a concise canonical company name"
                    ),
                },
            ).rename(columns={"document_id": "nda_document_id"})
            nda_records["nda_category"] = nda_records["nda_category"].map(
                lambda value: normalize_enum(value, NDA_CATEGORIES)
            )
            nda_records["party_name"] = nda_records["party_name"].map(
                clean_optional_text
            )
            nda_records["party_key"] = nda_records["party_key"].map(
                normalize_company_name
            )
            nda_records = nda_records.dropna(
                subset=["nda_category", "party_name", "party_key"]
            )[
                ["nda_document_id", "nda_category", "party_name", "party_key"]
            ].reset_index(drop=True)
            step.set_output(nda_records)

        totals = (
            nda_records.groupby("nda_category", sort=False)["nda_document_id"]
            .nunique()
            .rename("total_agreement_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(NDA category, distinct total agreement count)",
            len(nda_records),
            len(totals),
            output=totals,
        )

        audit_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS AUDITS",
            None,
            len(audit_documents),
            output=audit_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-23 auditor consent naming audit client)",
            input_rows=len(audit_documents),
        ) as step:
            audits = audit_documents.sem_filter(
                "The document {text} is an EX-23 auditor consent letter that names its "
                "audit-client company."
            ).reset_index(drop=True)
            step.set_output(audits)

        with tracker.step(
            "SEM_EXTRACT(audit-client company and canonical key)",
            input_rows=len(audits),
        ) as step:
            audit_records = audits.sem_extract(
                input_cols=["text"],
                output_cols={
                    "audit_client_company": (
                        "the audit-client company name stated in the consent letter"
                    ),
                    "audit_client_key": (
                        "that audit-client company as a concise canonical company name"
                    ),
                },
            )
            audit_records["audit_client_company"] = audit_records[
                "audit_client_company"
            ].map(clean_optional_text)
            audit_records["audit_client_key"] = audit_records[
                "audit_client_key"
            ].map(normalize_company_name)
            audit_records = audit_records.dropna(
                subset=["audit_client_company", "audit_client_key"]
            )[
                ["audit_client_company", "audit_client_key"]
            ].reset_index(drop=True)
            step.set_output(audit_records)

        audit_matches = nda_records.merge(
            audit_records,
            left_on="party_key",
            right_on="audit_client_key",
            how="inner",
        )
        tracker.record(
            "JOIN(NDAS.party_key = AUDITS.audit_client_key)",
            {"NDAS": len(nda_records), "AUDITS": len(audit_records)},
            len(audit_matches),
            output=audit_matches,
        )

        linked = (
            audit_matches.groupby("nda_category", sort=False)
            .agg(
                audit_linked_agreement_count=("nda_document_id", "nunique"),
                matched_audit_client_companies=(
                    "audit_client_company",
                    _sorted_unique,
                ),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(NDA category, linked count, matched audit clients)",
            len(audit_matches),
            len(linked),
            output=linked,
        )

        joined = totals.merge(linked, on="nda_category", how="inner")
        tracker.record(
            "JOIN(TOTALS.nda_category = LINKED.nda_category)",
            {"TOTALS": len(totals), "LINKED": len(linked)},
            len(joined),
            output=joined,
        )

        projected = joined[
            [
                "nda_category",
                "total_agreement_count",
                "audit_linked_agreement_count",
                "matched_audit_client_companies",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(category, totals, audit-linked count, matched companies)",
            len(joined),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
