#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-090."""

from __future__ import annotations

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-090"
DATASET = "contract-exhibit"
EXCEPTION_COLUMNS = [
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "written_consent",
]


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def duration_comparison(row: pd.Series) -> str:
    employee = row["employee_non_solicitation_duration"]
    standstill = row["standstill_duration"]
    if employee > standstill:
        return "employee_restriction_longer"
    if employee == standstill:
        return "equal"
    return "standstill_longer"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a non-disclosure agreement, "
                "including a dual-labelled NDA exhibit."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        restriction_plan = memory_dataset(
            f"{TASK_ID}-restrictions", scoped
        ).sem_filter(
            filter=(
                "Keep the agreement only if it expressly contains both an operative "
                "standstill provision and an operative employee non-solicitation clause."
            ),
            depends_on=["text"],
        )
        started = time.time()
        restriction_result = restriction_plan.run(config)
        restricted = result_frame(restriction_result, scoped)
        tracker.record_semantic(
            "sem_filter", len(scoped), restricted, restriction_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", restricted
        ).sem_map(
            cols=[
                {"name": "standstill_duration", "type": float | None,
                 "desc": "The standstill duration normalized to months, or null when unstated or not quantifiable."},
                {"name": "employee_non_solicitation_duration", "type": float | None,
                 "desc": "The employee non-solicitation duration normalized to months, or null when unstated or not quantifiable."},
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
                {"name": "written_consent", "type": bool,
                 "desc": "True if an operative written-consent exception is present; otherwise false."},
            ],
            desc=(
                "Extract both restriction durations and determine the six named "
                "confidentiality-exception flags."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = [
            "standstill_duration",
            "employee_non_solicitation_duration",
            *EXCEPTION_COLUMNS,
        ]
        extracted = result_frame(extraction_result, restricted, generated)
        for column in ("standstill_duration", "employee_non_solicitation_duration"):
            extracted[column] = extracted[column].map(normalize_number)
        for column in EXCEPTION_COLUMNS:
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(restricted), extracted, extraction_result,
            time.time() - started,
        )

        complete = extracted.loc[
            extracted["standstill_duration"].notna()
            & extracted["employee_non_solicitation_duration"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), complete)

        projected = pd.DataFrame(
            {
                "class_label": complete.apply(duration_comparison, axis=1),
                "exception_count": complete[EXCEPTION_COLUMNS].sum(axis=1),
            }
        ).reset_index(drop=True)
        tracker.record("project", len(complete), projected)

        grouped = (
            projected.groupby("class_label", sort=False)
            .agg(
                agreement_count=("class_label", "size"),
                average_exception_count=("exception_count", "mean"),
            )
            .reset_index()
        )
        grouped["average_exception_count"] = grouped[
            "average_exception_count"
        ].round(2)
        tracker.record("groupby", len(projected), grouped)

        ordered = grouped.sort_values(
            ["agreement_count", "class_label"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
