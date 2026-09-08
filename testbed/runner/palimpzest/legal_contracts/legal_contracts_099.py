#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-099."""

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
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-099"
DATASET = "contract-exhibit"
NDA_CATEGORIES = {
    "mutual_nda",
    "unilateral_nda",
    "confidentiality_standstill",
}
LEGAL_SUFFIXES = {
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "limited", "llc", "llp", "lp", "ltd", "plc",
}


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_label(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or None


def normalize_category(value) -> str | None:
    normalized = normalize_label(value)
    aliases = {
        "mutual": "mutual_nda",
        "unilateral": "unilateral_nda",
        "confidentiality_and_standstill": "confidentiality_standstill",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in NDA_CATEGORIES else None


def normalize_company_name(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    tokens = re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def sorted_unique(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value) and str(value)})


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        nda_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, nda_documents)

        nda_filter_plan = memory_dataset(
            f"{TASK_ID}-ndas", nda_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is a mutual NDA, unilateral NDA, or "
                "confidentiality-and-standstill agreement, including a dual-status "
                "SEC exhibit."
            ),
            depends_on=["text"],
        )
        started = time.time()
        nda_filter_result = nda_filter_plan.run(config)
        ndas = result_frame(nda_filter_result, nda_documents)
        tracker.record_semantic(
            "sem_filter", len(nda_documents), ndas,
            nda_filter_result, time.time() - started,
        )

        nda_extraction_plan = memory_dataset(
            f"{TASK_ID}-nda-extraction", ndas
        ).sem_map(
            cols=[
                {"name": "nda_category", "type": str,
                 "desc": "Exactly mutual_nda, unilateral_nda, or confidentiality_standstill."},
                {"name": "party_name", "type": str,
                 "desc": "The stated corporate party to the agreement."},
                {"name": "party_key", "type": str,
                 "desc": "A concise canonical company name for that corporate party."},
            ],
            desc="Extract the NDA category and corporate-party identity.",
            depends_on=["text"],
        )
        started = time.time()
        nda_extraction_result = nda_extraction_plan.run(config)
        nda_records = result_frame(
            nda_extraction_result,
            ndas,
            ["nda_category", "party_name", "party_key"],
        ).rename(columns={"document_id": "nda_document_id"})
        nda_records["nda_category"] = nda_records["nda_category"].map(
            normalize_category
        )
        nda_records["party_name"] = nda_records["party_name"].map(normalize_text)
        nda_records["party_key"] = nda_records["party_key"].map(
            normalize_company_name
        )
        nda_records = nda_records.dropna(
            subset=["nda_category", "party_name", "party_key"]
        )[["nda_document_id", "nda_category", "party_name", "party_key"]].reset_index(
            drop=True
        )
        tracker.record_semantic(
            "sem_map", len(ndas), nda_records,
            nda_extraction_result, time.time() - started,
        )

        totals = (
            nda_records.groupby("nda_category", sort=False)["nda_document_id"]
            .nunique()
            .rename("total_agreement_count")
            .reset_index()
        )
        tracker.record("groupby", len(nda_records), totals)

        audit_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, audit_documents)

        audit_filter_plan = memory_dataset(
            f"{TASK_ID}-audits", audit_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-23 auditor consent letter "
                "that names its audit-client company."
            ),
            depends_on=["text"],
        )
        started = time.time()
        audit_filter_result = audit_filter_plan.run(config)
        audits = result_frame(audit_filter_result, audit_documents)
        tracker.record_semantic(
            "sem_filter", len(audit_documents), audits,
            audit_filter_result, time.time() - started,
        )

        audit_extraction_plan = memory_dataset(
            f"{TASK_ID}-audit-extraction", audits
        ).sem_map(
            cols=[
                {"name": "audit_client_company", "type": str,
                 "desc": "The audit-client company name stated in the consent letter."},
                {"name": "audit_client_key", "type": str,
                 "desc": "A concise canonical company name for that audit client."},
            ],
            desc="Extract the stated audit-client company and canonical key.",
            depends_on=["text"],
        )
        started = time.time()
        audit_extraction_result = audit_extraction_plan.run(config)
        audit_records = result_frame(
            audit_extraction_result,
            audits,
            ["audit_client_company", "audit_client_key"],
        )
        audit_records["audit_client_company"] = audit_records[
            "audit_client_company"
        ].map(normalize_text)
        audit_records["audit_client_key"] = audit_records["audit_client_key"].map(
            normalize_company_name
        )
        audit_records = audit_records.dropna(
            subset=["audit_client_company", "audit_client_key"]
        )[["audit_client_company", "audit_client_key"]].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(audits), audit_records,
            audit_extraction_result, time.time() - started,
        )

        audit_matches = nda_records.merge(
            audit_records,
            left_on="party_key",
            right_on="audit_client_key",
            how="inner",
        )
        tracker.record(
            "join",
            {"left": len(nda_records), "right": len(audit_records)},
            audit_matches,
        )

        linked = (
            audit_matches.groupby("nda_category", sort=False)
            .agg(
                audit_linked_agreement_count=("nda_document_id", "nunique"),
                matched_audit_client_companies=("audit_client_company", sorted_unique),
            )
            .reset_index()
        )
        tracker.record("groupby", len(audit_matches), linked)

        joined = totals.merge(linked, on="nda_category", how="inner")
        tracker.record(
            "join", {"left": len(totals), "right": len(linked)}, joined
        )

        projected = joined[
            [
                "nda_category",
                "total_agreement_count",
                "audit_linked_agreement_count",
                "matched_audit_client_companies",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(joined), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
