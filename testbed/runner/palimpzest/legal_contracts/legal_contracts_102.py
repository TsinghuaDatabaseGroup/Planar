#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-102."""

from __future__ import annotations

import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_mixed_documents,
    memory_dataset,
    normalize_scalar_value,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-102"
DATASET = "contract-exhibit"
EXCEPTION_COLUMNS = [
    "public_information_exception",
    "prior_knowledge_exception",
    "independent_development_exception",
    "third_party_receipt_exception",
    "legal_compulsion_exception",
    "consent_exception",
]
LEGAL_SUFFIXES = {
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "limited", "llc", "llp", "lp", "ltd", "plc",
}


def normalize_company_name(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None or pd.isna(value):
        return None
    tokens = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        nda_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, nda_documents)

        nda_filter_plan = memory_dataset(
            f"{TASK_ID}-delaware-ndas", nda_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it has non-disclosure-agreement status, "
                "including a dual-status SEC exhibit, and expressly states Delaware "
                "as a governing-law jurisdiction."
            ),
            depends_on=["text"],
        )
        started = time.time()
        nda_filter_result = nda_filter_plan.run(config)
        delaware_ndas = result_frame(nda_filter_result, nda_documents)
        tracker.record_semantic(
            "sem_filter", len(nda_documents), delaware_ndas,
            nda_filter_result, time.time() - started,
        )

        nda_extraction_plan = memory_dataset(
            f"{TASK_ID}-nda-extraction", delaware_ndas
        ).sem_map(
            cols=[
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical company key for the stated corporate party."},
                {"name": "public_information_exception", "type": bool,
                 "desc": "True only if public information is expressly excluded from confidential information; otherwise false."},
                {"name": "prior_knowledge_exception", "type": bool,
                 "desc": "True only if prior knowledge is expressly excluded from confidential information; otherwise false."},
                {"name": "independent_development_exception", "type": bool,
                 "desc": "True only if independent development is expressly excluded from confidential information; otherwise false."},
                {"name": "third_party_receipt_exception", "type": bool,
                 "desc": "True only if lawful third-party receipt is expressly excluded from confidential information; otherwise false."},
                {"name": "legal_compulsion_exception", "type": bool,
                 "desc": "True only if disclosure under legal compulsion is expressly permitted; otherwise false."},
                {"name": "consent_exception", "type": bool,
                 "desc": "True only if disclosure with consent is expressly permitted; otherwise false."},
            ],
            desc="Extract the corporate-party key and six confidentiality-exception flags.",
            depends_on=["text"],
        )
        started = time.time()
        nda_extraction_result = nda_extraction_plan.run(config)
        nda_records = result_frame(
            nda_extraction_result,
            delaware_ndas,
            ["company_key", *EXCEPTION_COLUMNS],
        ).rename(columns={"document_id": "nda_document_id"})
        nda_records["company_key"] = nda_records["company_key"].map(
            normalize_company_name
        )
        for column in EXCEPTION_COLUMNS:
            nda_records[column] = nda_records[column].map(parse_bool)
        nda_records = nda_records.dropna(subset=["company_key"])[
            ["nda_document_id", "company_key", *EXCEPTION_COLUMNS]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(delaware_ndas), nda_records,
            nda_extraction_result, time.time() - started,
        )

        certification_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, certification_documents)

        certification_filter_plan = memory_dataset(
            f"{TASK_ID}-cert302", certification_documents
        ).sem_filter(
            filter="Keep the document only if it is an EX-31 SOX Section 302 certification.",
            depends_on=["text"],
        )
        started = time.time()
        certification_filter_result = certification_filter_plan.run(config)
        certifications = result_frame(
            certification_filter_result, certification_documents
        )
        tracker.record_semantic(
            "sem_filter", len(certification_documents), certifications,
            certification_filter_result, time.time() - started,
        )

        certification_extraction_plan = memory_dataset(
            f"{TASK_ID}-cert302-extraction", certifications
        ).sem_map(
            cols=[
                {"name": "company_key", "type": str,
                 "desc": "A concise canonical key for the company named in the certification."}
            ],
            desc="Extract the canonical certified-company key.",
            depends_on=["text"],
        )
        started = time.time()
        certification_extraction_result = certification_extraction_plan.run(config)
        certification_keys = result_frame(
            certification_extraction_result, certifications, ["company_key"]
        )
        certification_keys["company_key"] = certification_keys["company_key"].map(
            normalize_company_name
        )
        certification_keys = certification_keys.dropna(subset=["company_key"])[
            ["company_key"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(certifications), certification_keys,
            certification_extraction_result, time.time() - started,
        )

        matched = nda_records.merge(certification_keys, on="company_key", how="inner")
        tracker.record(
            "join",
            {"left": len(nda_records), "right": len(certification_keys)},
            matched,
        )

        grouped = (
            matched.groupby(EXCEPTION_COLUMNS, sort=False, dropna=False)
            .agg(
                nda_document_count=("nda_document_id", "nunique"),
                distinct_matched_company_count=("company_key", "nunique"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(matched), grouped)

        filtered = grouped.loc[grouped["nda_document_count"].ge(2)].reset_index(
            drop=True
        )
        tracker.record("filter", len(grouped), filtered)
        answer = df_records(filtered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
