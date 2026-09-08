#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-089."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-089"
DATASET = "contract-exhibit"
EXCEPTION_COLUMNS = [
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
]
BUCKET_ORDER = {
    "high_5_to_6": 1,
    "medium_3_to_4": 2,
    "low_0_to_2": 3,
}


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_title(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()) or None


def coverage_bucket(value: float) -> str:
    if value >= 5:
        return "high_5_to_6"
    if value >= 3:
        return "medium_3_to_4"
    return "low_0_to_2"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if its exhibit type is exactly NDA, not a "
                "dual-labelled exhibit, and it has an identifiable agreement title."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        definition_plan = memory_dataset(
            f"{TASK_ID}-definition", scoped
        ).sem_filter(
            filter=(
                "Keep the agreement only if it explicitly defines confidential "
                "information."
            ),
            depends_on=["text"],
        )
        started = time.time()
        definition_result = definition_plan.run(config)
        defined = result_frame(definition_result, scoped)
        tracker.record_semantic(
            "sem_filter", len(scoped), defined, definition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", defined
        ).sem_map(
            cols=[
                {"name": "agreement_title", "type": str,
                 "desc": "The identifiable agreement title stated in the document."},
                {"name": "public_information", "type": bool,
                 "desc": "True if an operative public-information exception is present; otherwise false."},
                {"name": "prior_knowledge", "type": bool,
                 "desc": "True if an operative prior-knowledge exception is present; otherwise false."},
                {"name": "independent_development", "type": bool,
                 "desc": "True if an operative independent-development exception is present; otherwise false."},
                {"name": "third_party_receipt", "type": bool,
                 "desc": "True if an operative lawful third-party-receipt exception is present; otherwise false."},
                {"name": "legal_compulsion", "type": bool,
                 "desc": "True if an operative legal-compulsion exception is present; otherwise false."},
                {"name": "consent", "type": bool,
                 "desc": "True if an operative disclosure-with-consent exception is present; otherwise false."},
            ],
            desc="Extract the agreement title and six standard exception flags.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            defined,
            ["agreement_title", *EXCEPTION_COLUMNS],
        )
        extracted["agreement_title"] = extracted["agreement_title"].map(normalize_text)
        for column in EXCEPTION_COLUMNS:
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(defined), extracted, extraction_result, time.time() - started
        )

        projected = pd.DataFrame(
            {
                "normalized_agreement_title": extracted["agreement_title"].map(
                    normalize_title
                ),
                "exception_count": extracted[EXCEPTION_COLUMNS].sum(axis=1),
            }
        ).reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        title_families = (
            projected.groupby("normalized_agreement_title", sort=False)
            .agg(
                family_size=("normalized_agreement_title", "size"),
                average_exception_count=("exception_count", "mean"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(projected), title_families)

        large_families = title_families.loc[
            title_families["family_size"].ge(3)
        ].reset_index(drop=True)
        tracker.record("filter", len(title_families), large_families)

        bucketed = pd.DataFrame(
            {
                "coverage_bucket": large_families["average_exception_count"].map(
                    coverage_bucket
                ),
                "family_size": large_families["family_size"],
            }
        ).reset_index(drop=True)
        tracker.record("project", len(large_families), bucketed)

        grouped = (
            bucketed.groupby("coverage_bucket", sort=False)
            .agg(
                family_count=("coverage_bucket", "size"),
                average_family_size=("family_size", "mean"),
            )
            .reset_index()
        )
        grouped["average_family_size"] = grouped["average_family_size"].round(2)
        tracker.record("groupby", len(bucketed), grouped)

        ordered = grouped.sort_values(
            "coverage_bucket",
            key=lambda values: values.map(BUCKET_ORDER),
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
