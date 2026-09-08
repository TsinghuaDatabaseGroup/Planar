#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-085."""

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
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-085"
DATASET = "contract-exhibit"
FILING_FAMILIES = {
    "insider_trading",
    "clawback",
    "auditor_consent",
    "subsidiary_list",
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


def normalize_family(value) -> str | None:
    normalized = normalize_label(value)
    aliases = {
        "insider_trading_policy": "insider_trading",
        "clawback_policy": "clawback",
        "auditor_consent_letter": "auditor_consent",
        "subsidiary_list_filing": "subsidiary_list",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in FILING_FAMILIES else None


def normalize_optional_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        if value in {0, 1}:
            return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def normalize_integer(value) -> int | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else int(number)


def sorted_distinct(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value) and str(value)})


def any_true(values: pd.Series) -> bool:
    return any(value is True for value in values)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        sox_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, sox_documents)

        sox_filter_plan = memory_dataset(
            f"{TASK_ID}-sox", sox_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is a genuine Sarbanes-Oxley "
                "Section 302 certification."
            ),
            depends_on=["text"],
        )
        started = time.time()
        sox_filter_result = sox_filter_plan.run(config)
        sox_certifications = result_frame(sox_filter_result, sox_documents)
        tracker.record_semantic(
            "sem_filter", len(sox_documents), sox_certifications,
            sox_filter_result, time.time() - started,
        )

        sox_extraction_plan = memory_dataset(
            f"{TASK_ID}-sox-extraction", sox_certifications
        ).sem_map(
            cols=[
                {
                    "name": "sox_company",
                    "type": str,
                    "desc": "The company entity certified by the document.",
                },
                {
                    "name": "certifying_officer",
                    "type": str,
                    "desc": (
                        "The certifying officer's full name together with the stated title."
                    ),
                },
            ],
            desc="Extract the certified company and certifying officer with title.",
            depends_on=["text"],
        )
        started = time.time()
        sox_extraction_result = sox_extraction_plan.run(config)
        sox_records = result_frame(
            sox_extraction_result,
            sox_certifications,
            ["sox_company", "certifying_officer"],
        )
        sox_records["sox_company"] = sox_records["sox_company"].map(normalize_text)
        sox_records["certifying_officer"] = sox_records[
            "certifying_officer"
        ].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(sox_certifications), sox_records,
            sox_extraction_result, time.time() - started,
        )

        sox_groups = (
            sox_records.groupby("sox_company", sort=False)
            .agg(certifying_officers=("certifying_officer", sorted_distinct))
            .reset_index()
        )
        tracker.record("groupby", len(sox_records), sox_groups)

        other_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, other_documents)

        other_filter_plan = memory_dataset(
            f"{TASK_ID}-other", other_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an insider-trading policy, a "
                "clawback policy, an independent auditor consent letter, or a "
                "subsidiary-list filing."
            ),
            depends_on=["text"],
        )
        started = time.time()
        other_filter_result = other_filter_plan.run(config)
        other_filings = result_frame(other_filter_result, other_documents)
        tracker.record_semantic(
            "sem_filter", len(other_documents), other_filings,
            other_filter_result, time.time() - started,
        )

        other_extraction_plan = memory_dataset(
            f"{TASK_ID}-other-extraction", other_filings
        ).sem_map(
            cols=[
                {
                    "name": "other_company",
                    "type": str,
                    "desc": "The company entity to which the filing applies.",
                },
                {
                    "name": "filing_family",
                    "type": str,
                    "desc": (
                        "Exactly insider_trading, clawback, auditor_consent, or "
                        "subsidiary_list."
                    ),
                },
                {
                    "name": "auditor_firm",
                    "type": str | None,
                    "desc": (
                        "The auditor firm for an auditor-consent filing, or null when "
                        "not applicable or unstated."
                    ),
                },
                {
                    "name": "misconduct_trigger",
                    "type": bool | None,
                    "desc": (
                        "For a clawback policy, true if misconduct triggers recovery "
                        "and false if it does not; null for other filing families."
                    ),
                },
                {
                    "name": "covered_person_category",
                    "type": str | None,
                    "desc": (
                        "For a clawback policy, a concise normalized covered-person "
                        "category; null for other filing families or when unstated."
                    ),
                },
                {
                    "name": "subsidiary_count",
                    "type": int | None,
                    "desc": (
                        "For a subsidiary-list filing, the number of listed subsidiary "
                        "legal entities; null for other filing families."
                    ),
                },
            ],
            desc=(
                "Classify each filing family and extract its company plus the "
                "family-specific auditor, clawback, or subsidiary fields."
            ),
            depends_on=["text"],
        )
        started = time.time()
        other_extraction_result = other_extraction_plan.run(config)
        generated = [
            "other_company",
            "filing_family",
            "auditor_firm",
            "misconduct_trigger",
            "covered_person_category",
            "subsidiary_count",
        ]
        other_records = result_frame(other_extraction_result, other_filings, generated)
        other_records["other_company"] = other_records["other_company"].map(
            normalize_text
        )
        other_records["filing_family"] = other_records["filing_family"].map(
            normalize_family
        )
        other_records["auditor_firm"] = other_records["auditor_firm"].map(
            normalize_text
        )
        other_records["misconduct_trigger"] = other_records[
            "misconduct_trigger"
        ].map(normalize_optional_bool)
        other_records["covered_person_category"] = other_records[
            "covered_person_category"
        ].map(normalize_label)
        other_records["subsidiary_count"] = other_records["subsidiary_count"].map(
            normalize_integer
        )
        tracker.record_semantic(
            "sem_map", len(other_filings), other_records,
            other_extraction_result, time.time() - started,
        )

        other_groups = (
            other_records.groupby("other_company", sort=False)
            .agg(
                matched_families=("filing_family", sorted_distinct),
                auditor_firms=("auditor_firm", sorted_distinct),
                has_misconduct_trigger=("misconduct_trigger", any_true),
                covered_person_categories=(
                    "covered_person_category",
                    sorted_distinct,
                ),
                largest_subsidiary_count=("subsidiary_count", "max"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(other_records), other_groups)

        qualifying_other = other_groups.loc[
            other_groups["matched_families"].map(
                lambda values: FILING_FAMILIES.issubset(set(values))
            )
            & other_groups["has_misconduct_trigger"].eq(True)
            & other_groups["largest_subsidiary_count"].notna()
            & other_groups["largest_subsidiary_count"].ge(30)
        ].reset_index(drop=True)
        tracker.record("filter", len(other_groups), qualifying_other)

        join_columns = [
            "sox_company",
            "certifying_officers",
            "other_company",
            "matched_families",
            "auditor_firms",
            "has_misconduct_trigger",
            "covered_person_categories",
            "largest_subsidiary_count",
        ]
        if sox_groups.empty or qualifying_other.empty:
            matched = pd.DataFrame(columns=join_columns)
            tracker.record(
                "sem_join",
                {"left": len(sox_groups), "right": len(qualifying_other)},
                matched,
            )
        else:
            join_plan = memory_dataset(
                f"{TASK_ID}-sox-groups", sox_groups
            ).sem_join(
                memory_dataset(f"{TASK_ID}-other-groups", qualifying_other),
                condition=(
                    "The SOX-certified company and the company represented by the "
                    "other filings are the same corporate entity despite differences "
                    "in naming, capitalization, punctuation, or corporate suffixes."
                ),
                depends_on=["sox_company", "other_company"],
            )
            started = time.time()
            join_result = join_plan.run(config)
            matched = result_frame(join_result)
            if matched.empty:
                matched = pd.DataFrame(columns=join_columns)
            tracker.record_semantic(
                "sem_join",
                {"left": len(sox_groups), "right": len(qualifying_other)},
                matched,
                join_result,
                time.time() - started,
            )

        projected = matched.rename(columns={"sox_company": "company"})[
            [
                "company",
                "certifying_officers",
                "auditor_firms",
                "covered_person_categories",
                "largest_subsidiary_count",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(matched), projected)

        ordered = projected.sort_values(
            ["largest_subsidiary_count", "company"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(projected), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
