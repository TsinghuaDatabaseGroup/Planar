#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-088."""

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

TASK_ID = "legal_contracts-088"
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
            filter="Keep the document only if it is a genuine EX-31 SOX officer certification.",
            depends_on=["text"],
        )
        started = time.time()
        sox_filter_result = sox_filter_plan.run(config)
        certifications = result_frame(sox_filter_result, sox_documents)
        tracker.record_semantic(
            "sem_filter", len(sox_documents), certifications,
            sox_filter_result, time.time() - started,
        )

        sox_extraction_plan = memory_dataset(
            f"{TASK_ID}-sox-extraction", certifications
        ).sem_map(
            cols=[
                {"name": "sox_company", "type": str,
                 "desc": "The company entity certified by the document."},
                {"name": "certifying_officer", "type": str,
                 "desc": "The certifying officer's full name together with the stated title."},
            ],
            desc="Extract the certified company and certifying officer with title.",
            depends_on=["text"],
        )
        started = time.time()
        sox_extraction_result = sox_extraction_plan.run(config)
        sox_records = result_frame(
            sox_extraction_result,
            certifications,
            ["sox_company", "certifying_officer"],
        )
        sox_records["sox_company"] = sox_records["sox_company"].map(normalize_text)
        sox_records["certifying_officer"] = sox_records[
            "certifying_officer"
        ].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(certifications), sox_records,
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
            filter="Keep the document only if it is a genuine EX-97 clawback policy.",
            depends_on=["text"],
        )
        started = time.time()
        clawback_filter_result = clawback_filter_plan.run(config)
        clawback_policies = result_frame(clawback_filter_result, clawback_documents)
        tracker.record_semantic(
            "sem_filter", len(clawback_documents), clawback_policies,
            clawback_filter_result, time.time() - started,
        )

        trigger_filter_plan = memory_dataset(
            f"{TASK_ID}-dual-trigger", clawback_policies
        ).sem_filter(
            filter=(
                "Keep the policy only if it provides recovery for a financial "
                "restatement and also permits recovery for misconduct; both triggers "
                "must be present."
            ),
            depends_on=["text"],
        )
        started = time.time()
        trigger_filter_result = trigger_filter_plan.run(config)
        dual_trigger = result_frame(trigger_filter_result, clawback_policies)
        tracker.record_semantic(
            "sem_filter", len(clawback_policies), dual_trigger,
            trigger_filter_result, time.time() - started,
        )

        clawback_extraction_plan = memory_dataset(
            f"{TASK_ID}-clawback-extraction", dual_trigger
        ).sem_map(
            cols=[
                {"name": "clawback_company", "type": str,
                 "desc": "The company entity covered by the recovery policy."},
                {"name": "covered_person_category", "type": str,
                 "desc": "A concise normalized category for the people covered by the recovery policy."},
            ],
            desc="Extract the policy company and normalized covered-person category.",
            depends_on=["text"],
        )
        started = time.time()
        clawback_extraction_result = clawback_extraction_plan.run(config)
        clawback_records = result_frame(
            clawback_extraction_result,
            dual_trigger,
            ["clawback_company", "covered_person_category"],
        )
        clawback_records["clawback_company"] = clawback_records[
            "clawback_company"
        ].map(normalize_text)
        clawback_records["covered_person_category"] = clawback_records[
            "covered_person_category"
        ].map(normalize_label)
        tracker.record_semantic(
            "sem_map", len(dual_trigger), clawback_records,
            clawback_extraction_result, time.time() - started,
        )

        clawback_distinct = clawback_records[
            ["clawback_company", "covered_person_category"]
        ].drop_duplicates(ignore_index=True)
        tracker.record("dedup", len(clawback_records), clawback_distinct)

        join_columns = [
            "sox_company",
            "certifying_officers",
            "clawback_company",
            "covered_person_category",
        ]
        if sox_groups.empty or clawback_distinct.empty:
            matched = pd.DataFrame(columns=join_columns)
            tracker.record(
                "sem_join",
                {"left": len(sox_groups), "right": len(clawback_distinct)},
                matched,
            )
        else:
            join_plan = memory_dataset(
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
            join_result = join_plan.run(config)
            matched = result_frame(join_result)
            if matched.empty:
                matched = pd.DataFrame(columns=join_columns)
            tracker.record_semantic(
                "sem_join",
                {"left": len(sox_groups), "right": len(clawback_distinct)},
                matched,
                join_result,
                time.time() - started,
            )

        projected = matched.rename(columns={"sox_company": "company"})[
            ["company", "certifying_officers", "covered_person_category"]
        ].reset_index(drop=True)
        tracker.record("project", len(matched), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
