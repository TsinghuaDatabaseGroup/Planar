#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-076."""

from __future__ import annotations

import ast
import json
import os
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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-076"
DATASET = "contract-exhibit"
PERIOD_TYPES = ("annual", "quarterly", "other")
JOIN_KEYS = ["company_key", "reporting_period_type", "reporting_period_end_date"]


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_iso_date(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def normalize_string_list(value) -> list[str]:
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return []
    parsed = value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value.strip())
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
    if pd.api.types.is_list_like(parsed) and not isinstance(parsed, dict):
        parsed = list(parsed)
    else:
        parsed = [parsed]
    return [
        normalized
        for item in parsed
        if (normalized := " ".join(str(item).strip().split()))
    ]


def lexicographic_min(values) -> str | None:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    return min(cleaned) if cleaned else None


def sorted_values(values) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value)})


def sorted_distinct_flatten(values) -> list[str]:
    return sorted(
        {
            str(item)
            for value in values
            for item in value
            if item is not None and str(item)
        }
    )


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        section_302_docs = load_mixed_documents(DATASET)
        tracker.record("scan", None, section_302_docs)

        section_302_filter_plan = memory_dataset(
            f"{TASK_ID}-302-filter", section_302_docs
        ).sem_filter(
            filter=(
                "Keep the document only if it is a genuine Sarbanes-Oxley "
                "Section 302 certification."
            ),
            depends_on=["text"],
        )
        started = time.time()
        section_302_filter_result = section_302_filter_plan.run(config)
        section_302_certifications = result_frame(
            section_302_filter_result, section_302_docs
        )
        tracker.record_semantic(
            "sem_filter",
            len(section_302_docs),
            section_302_certifications,
            section_302_filter_result,
            time.time() - started,
        )

        section_302_extraction_plan = memory_dataset(
            f"{TASK_ID}-302-extraction", section_302_certifications
        ).sem_map(
            cols=[
                {
                    "name": "company_key",
                    "type": str | None,
                    "desc": "A stable normalized identity key for the stated company.",
                },
                {
                    "name": "company_name_variant",
                    "type": str | None,
                    "desc": "The company name exactly as stated in this certification.",
                },
                {
                    "name": "reporting_period_type",
                    "type": str | None,
                    "desc": "The explicitly stated period type as annual, quarterly, or other; null when unstated.",
                },
                {
                    "name": "reporting_period_end_date",
                    "type": str | None,
                    "desc": "The explicitly stated reporting-period end date as YYYY-MM-DD; null when unstated.",
                },
                {
                    "name": "certifying_officers",
                    "type": list[str],
                    "desc": "The full names of all officers who certify the document.",
                },
            ],
            desc=(
                "Extract the company identity, reporting period, and certifying "
                "officers from the Section 302 certification."
            ),
            depends_on=["text"],
        )
        started = time.time()
        section_302_extraction_result = section_302_extraction_plan.run(config)
        generated = [
            "company_key",
            "company_name_variant",
            "reporting_period_type",
            "reporting_period_end_date",
            "certifying_officers",
        ]
        section_302_extracted = result_frame(
            section_302_extraction_result,
            section_302_certifications,
            generated,
        )
        section_302_extracted["company_key"] = section_302_extracted[
            "company_key"
        ].map(normalize_text)
        section_302_extracted["company_name_variant"] = section_302_extracted[
            "company_name_variant"
        ].map(normalize_text)
        section_302_extracted["reporting_period_type"] = section_302_extracted[
            "reporting_period_type"
        ].map(lambda value: normalize_enum(value, PERIOD_TYPES))
        section_302_extracted["reporting_period_end_date"] = section_302_extracted[
            "reporting_period_end_date"
        ].map(normalize_iso_date)
        section_302_extracted["certifying_officers"] = section_302_extracted[
            "certifying_officers"
        ].map(normalize_string_list)
        tracker.record_semantic(
            "sem_map",
            len(section_302_certifications),
            section_302_extracted,
            section_302_extraction_result,
            time.time() - started,
        )

        section_302_dated = section_302_extracted.loc[
            section_302_extracted["reporting_period_type"].notna()
            & section_302_extracted["reporting_period_end_date"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(section_302_extracted), section_302_dated)

        section_302_groups = (
            section_302_dated.groupby(JOIN_KEYS, sort=False, dropna=False)
            .agg(
                company_name_302=("company_name_variant", lexicographic_min),
                section_302_document_ids=("document_id", sorted_values),
                section_302_certifying_officers=(
                    "certifying_officers",
                    sorted_distinct_flatten,
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(section_302_dated), section_302_groups)

        section_906_docs = load_mixed_documents(DATASET)
        tracker.record("scan", None, section_906_docs)

        section_906_filter_plan = memory_dataset(
            f"{TASK_ID}-906-filter", section_906_docs
        ).sem_filter(
            filter=(
                "Keep the document only if it is a genuine Sarbanes-Oxley "
                "Section 906 certification."
            ),
            depends_on=["text"],
        )
        started = time.time()
        section_906_filter_result = section_906_filter_plan.run(config)
        section_906_certifications = result_frame(
            section_906_filter_result, section_906_docs
        )
        tracker.record_semantic(
            "sem_filter",
            len(section_906_docs),
            section_906_certifications,
            section_906_filter_result,
            time.time() - started,
        )

        section_906_extraction_plan = memory_dataset(
            f"{TASK_ID}-906-extraction", section_906_certifications
        ).sem_map(
            cols=[
                {
                    "name": "company_key",
                    "type": str | None,
                    "desc": "A stable normalized identity key for the stated company.",
                },
                {
                    "name": "company_name_variant",
                    "type": str | None,
                    "desc": "The company name exactly as stated in this certification.",
                },
                {
                    "name": "reporting_period_type",
                    "type": str | None,
                    "desc": "The explicitly stated period type as annual, quarterly, or other; null when unstated.",
                },
                {
                    "name": "reporting_period_end_date",
                    "type": str | None,
                    "desc": "The explicitly stated reporting-period end date as YYYY-MM-DD; null when unstated.",
                },
                {
                    "name": "certifying_officers",
                    "type": list[str],
                    "desc": "The full names of all officers who certify the document.",
                },
            ],
            desc=(
                "Extract the company identity, reporting period, and certifying "
                "officers from the Section 906 certification."
            ),
            depends_on=["text"],
        )
        started = time.time()
        section_906_extraction_result = section_906_extraction_plan.run(config)
        section_906_extracted = result_frame(
            section_906_extraction_result,
            section_906_certifications,
            generated,
        )
        section_906_extracted["company_key"] = section_906_extracted[
            "company_key"
        ].map(normalize_text)
        section_906_extracted["company_name_variant"] = section_906_extracted[
            "company_name_variant"
        ].map(normalize_text)
        section_906_extracted["reporting_period_type"] = section_906_extracted[
            "reporting_period_type"
        ].map(lambda value: normalize_enum(value, PERIOD_TYPES))
        section_906_extracted["reporting_period_end_date"] = section_906_extracted[
            "reporting_period_end_date"
        ].map(normalize_iso_date)
        section_906_extracted["certifying_officers"] = section_906_extracted[
            "certifying_officers"
        ].map(normalize_string_list)
        tracker.record_semantic(
            "sem_map",
            len(section_906_certifications),
            section_906_extracted,
            section_906_extraction_result,
            time.time() - started,
        )

        section_906_dated = section_906_extracted.loc[
            section_906_extracted["reporting_period_type"].notna()
            & section_906_extracted["reporting_period_end_date"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(section_906_extracted), section_906_dated)

        section_906_groups = (
            section_906_dated.groupby(JOIN_KEYS, sort=False, dropna=False)
            .agg(
                company_name_906=("company_name_variant", lexicographic_min),
                section_906_document_ids=("document_id", sorted_values),
                section_906_certifying_officers=(
                    "certifying_officers",
                    sorted_distinct_flatten,
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(section_906_dated), section_906_groups)

        joined = section_302_groups.dropna(subset=["company_key"]).merge(
            section_906_groups.dropna(subset=["company_key"]),
            on=JOIN_KEYS,
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "section_302_groups": len(section_302_groups),
                "section_906_groups": len(section_906_groups),
            },
            joined,
        )

        projected = joined.copy()
        projected["company_name"] = pd.Series(
            (
                lexicographic_min(
                    [row["company_name_302"], row["company_name_906"]]
                )
                for _, row in joined.iterrows()
            ),
            index=joined.index,
            dtype="object",
        )
        projected = projected[
            [
                "company_name",
                "reporting_period_type",
                "reporting_period_end_date",
                "section_302_document_ids",
                "section_302_certifying_officers",
                "section_906_document_ids",
                "section_906_certifying_officers",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(joined), projected)

        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
