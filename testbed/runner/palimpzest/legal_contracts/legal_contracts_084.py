#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-084."""

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

TASK_ID = "legal_contracts-084"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_label(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or None


def normalize_integer(value) -> int | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else int(number)


def sorted_distinct(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value) and str(value)})


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

        clawback_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, clawback_documents)

        clawback_filter_plan = memory_dataset(
            f"{TASK_ID}-clawback", clawback_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is a clawback or compensation-recovery "
                "policy."
            ),
            depends_on=["text"],
        )
        started = time.time()
        clawback_filter_result = clawback_filter_plan.run(config)
        clawback_policies = result_frame(clawback_filter_result, clawback_documents)
        tracker.record_semantic(
            "sem_filter", len(clawback_documents), clawback_policies,
            clawback_filter_result, time.time() - started,
        )

        misconduct_filter_plan = memory_dataset(
            f"{TASK_ID}-misconduct", clawback_policies
        ).sem_filter(
            filter="Keep the policy only if it permits compensation recovery for misconduct.",
            depends_on=["text"],
        )
        started = time.time()
        misconduct_filter_result = misconduct_filter_plan.run(config)
        misconduct_policies = result_frame(
            misconduct_filter_result, clawback_policies
        )
        tracker.record_semantic(
            "sem_filter", len(clawback_policies), misconduct_policies,
            misconduct_filter_result, time.time() - started,
        )

        clawback_extraction_plan = memory_dataset(
            f"{TASK_ID}-clawback-extraction", misconduct_policies
        ).sem_map(
            cols=[
                {
                    "name": "clawback_company",
                    "type": str,
                    "desc": "The company entity covered by the recovery policy.",
                },
                {
                    "name": "covered_person_category",
                    "type": str,
                    "desc": (
                        "A concise normalized category for the people covered by the "
                        "recovery policy."
                    ),
                },
            ],
            desc="Extract the policy company and covered-person category.",
            depends_on=["text"],
        )
        started = time.time()
        clawback_extraction_result = clawback_extraction_plan.run(config)
        clawback_records = result_frame(
            clawback_extraction_result,
            misconduct_policies,
            ["clawback_company", "covered_person_category"],
        )
        clawback_records["clawback_company"] = clawback_records[
            "clawback_company"
        ].map(normalize_text)
        clawback_records["covered_person_category"] = clawback_records[
            "covered_person_category"
        ].map(normalize_label)
        tracker.record_semantic(
            "sem_map", len(misconduct_policies), clawback_records,
            clawback_extraction_result, time.time() - started,
        )

        clawback_distinct = clawback_records[
            ["clawback_company", "covered_person_category"]
        ].drop_duplicates(ignore_index=True)
        tracker.record("dedup", len(clawback_records), clawback_distinct)

        subsidiary_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, subsidiary_documents)

        subsidiary_filter_plan = memory_dataset(
            f"{TASK_ID}-subsidiaries", subsidiary_documents
        ).sem_filter(
            filter="Keep the document only if it is a genuine subsidiary-list filing.",
            depends_on=["text"],
        )
        started = time.time()
        subsidiary_filter_result = subsidiary_filter_plan.run(config)
        subsidiary_filings = result_frame(
            subsidiary_filter_result, subsidiary_documents
        )
        tracker.record_semantic(
            "sem_filter", len(subsidiary_documents), subsidiary_filings,
            subsidiary_filter_result, time.time() - started,
        )

        subsidiary_extraction_plan = memory_dataset(
            f"{TASK_ID}-subsidiary-extraction", subsidiary_filings
        ).sem_map(
            cols=[
                {
                    "name": "subsidiary_company",
                    "type": str,
                    "desc": "The parent company entity identified by the filing.",
                },
                {
                    "name": "subsidiary_count",
                    "type": int,
                    "desc": "The number of listed subsidiary legal entities.",
                },
            ],
            desc="Extract the parent company and number of listed subsidiaries.",
            depends_on=["text"],
        )
        started = time.time()
        subsidiary_extraction_result = subsidiary_extraction_plan.run(config)
        subsidiary_records = result_frame(
            subsidiary_extraction_result,
            subsidiary_filings,
            ["subsidiary_company", "subsidiary_count"],
        )
        subsidiary_records["subsidiary_company"] = subsidiary_records[
            "subsidiary_company"
        ].map(normalize_text)
        subsidiary_records["subsidiary_count"] = subsidiary_records[
            "subsidiary_count"
        ].map(normalize_integer)
        tracker.record_semantic(
            "sem_map", len(subsidiary_filings), subsidiary_records,
            subsidiary_extraction_result, time.time() - started,
        )

        large_filings = subsidiary_records.loc[
            subsidiary_records["subsidiary_count"].notna()
            & subsidiary_records["subsidiary_count"].ge(70)
        ].reset_index(drop=True)
        tracker.record("filter", len(subsidiary_records), large_filings)

        subsidiary_groups = (
            large_filings.groupby("subsidiary_company", sort=False)
            .agg(largest_subsidiary_count=("subsidiary_count", "max"))
            .reset_index()
        )
        tracker.record("groupby", len(large_filings), subsidiary_groups)

        first_join_columns = [
            "sox_company",
            "certifying_officers",
            "clawback_company",
            "covered_person_category",
        ]
        if sox_groups.empty or clawback_distinct.empty:
            sox_clawback = pd.DataFrame(columns=first_join_columns)
            tracker.record(
                "sem_join",
                {"left": len(sox_groups), "right": len(clawback_distinct)},
                sox_clawback,
            )
        else:
            first_join_plan = memory_dataset(
                f"{TASK_ID}-sox-groups", sox_groups
            ).sem_join(
                memory_dataset(f"{TASK_ID}-clawback-groups", clawback_distinct),
                condition=(
                    "The SOX-certified company and the clawback-policy company are "
                    "the same corporate entity despite differences in naming, "
                    "capitalization, punctuation, or corporate suffixes."
                ),
                depends_on=["sox_company", "clawback_company"],
            )
            started = time.time()
            first_join_result = first_join_plan.run(config)
            sox_clawback = result_frame(first_join_result)
            if sox_clawback.empty:
                sox_clawback = pd.DataFrame(columns=first_join_columns)
            tracker.record_semantic(
                "sem_join",
                {"left": len(sox_groups), "right": len(clawback_distinct)},
                sox_clawback,
                first_join_result,
                time.time() - started,
            )

        final_join_columns = [
            *first_join_columns,
            "subsidiary_company",
            "largest_subsidiary_count",
        ]
        if sox_clawback.empty or subsidiary_groups.empty:
            matched = pd.DataFrame(columns=final_join_columns)
            tracker.record(
                "sem_join",
                {"left": len(sox_clawback), "right": len(subsidiary_groups)},
                matched,
            )
        else:
            final_join_plan = memory_dataset(
                f"{TASK_ID}-sox-clawback", sox_clawback
            ).sem_join(
                memory_dataset(
                    f"{TASK_ID}-subsidiary-groups", subsidiary_groups
                ),
                condition=(
                    "The previously matched company and the parent company in the "
                    "subsidiary filing are the same corporate entity despite "
                    "differences in naming, capitalization, punctuation, or "
                    "corporate suffixes."
                ),
                depends_on=["sox_company", "subsidiary_company"],
            )
            started = time.time()
            final_join_result = final_join_plan.run(config)
            matched = result_frame(final_join_result)
            if matched.empty:
                matched = pd.DataFrame(columns=final_join_columns)
            tracker.record_semantic(
                "sem_join",
                {"left": len(sox_clawback), "right": len(subsidiary_groups)},
                matched,
                final_join_result,
                time.time() - started,
            )

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
        tracker.record("orderby", len(matched), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
