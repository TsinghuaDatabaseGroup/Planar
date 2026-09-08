#!/usr/bin/env python3
"""Plan-optimization pipeline for legal_contracts-090."""

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
    pz,
    result_frame,
    run_plan_optimization,
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
DURATION_COLUMNS = [
    "standstill_duration",
    "employee_non_solicitation_duration",
]


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def has_stated_durations(record: dict) -> bool:
    return all(
        normalize_number(record.get(column)) is not None
        for column in DURATION_COLUMNS
    )


def duration_comparison(row: dict) -> str | None:
    employee = normalize_number(row.get("employee_non_solicitation_duration"))
    standstill = normalize_number(row.get("standstill_duration"))
    if employee is None or standstill is None:
        return None
    if employee > standstill:
        return "employee_restriction_longer"
    if employee == standstill:
        return "equal"
    return "standstill_longer"


def comparison_fields(record: dict) -> dict:
    return {
        "class_label": duration_comparison(record),
        "exception_count": sum(
            parse_bool(record.get(column)) for column in EXCEPTION_COLUMNS
        ),
    }


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a non-disclosure agreement, "
                "including a dual-labelled NDA exhibit."
            ),
            depends_on=["text"],
        )
        plan = plan.sem_filter(
            filter=(
                "Keep the agreement only if it expressly contains both an operative "
                "standstill provision and an operative employee non-solicitation clause."
            ),
            depends_on=["text"],
        )
        plan = plan.sem_map(
            cols=[
                {
                    "name": "standstill_duration",
                    "type": float | None,
                    "desc": (
                        "The standstill duration normalized to months, or null "
                        "when unstated or not quantifiable."
                    ),
                },
                {
                    "name": "employee_non_solicitation_duration",
                    "type": float | None,
                    "desc": (
                        "The employee non-solicitation duration normalized to "
                        "months, or null when unstated or not quantifiable."
                    ),
                },
                {
                    "name": "public_information",
                    "type": bool,
                    "desc": "True if an operative public-information exception is present; otherwise false.",
                },
                {
                    "name": "prior_knowledge",
                    "type": bool,
                    "desc": "True if an operative prior-knowledge exception is present; otherwise false.",
                },
                {
                    "name": "independent_development",
                    "type": bool,
                    "desc": "True if an operative independent-development exception is present; otherwise false.",
                },
                {
                    "name": "third_party_receipt",
                    "type": bool,
                    "desc": "True if an operative lawful third-party-receipt exception is present; otherwise false.",
                },
                {
                    "name": "legal_compulsion",
                    "type": bool,
                    "desc": "True if an operative legal-compulsion exception is present; otherwise false.",
                },
                {
                    "name": "written_consent",
                    "type": bool,
                    "desc": "True if an operative written-consent exception is present; otherwise false.",
                },
            ],
            desc=(
                "Extract both restriction durations and determine the six named "
                "confidentiality-exception flags."
            ),
            depends_on=["text"],
        )
        plan = plan.filter(
            has_stated_durations,
            depends_on=DURATION_COLUMNS,
        )
        plan = plan.map(
            comparison_fields,
            cols=[
                {
                    "name": "class_label",
                    "type": str | None,
                    "desc": "The comparison of the two restriction durations.",
                },
                {
                    "name": "exception_count",
                    "type": int,
                    "desc": "The number of the six named exceptions present.",
                },
            ],
            depends_on=[*DURATION_COLUMNS, *EXCEPTION_COLUMNS],
        )
        plan = plan.groupby(
            pz.GroupBySig(
                group_by_fields=["class_label"],
                agg_funcs=["count", "average"],
                agg_fields=["class_label", "exception_count"],
            )
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = result_frame(
            optimized.result,
            documents,
            ["class_label", "count(class_label)", "average(exception_count)"],
        ).rename(
            columns={
                "count(class_label)": "agreement_count",
                "average(exception_count)": "average_exception_count",
            }
        )
        output = output[
            ["class_label", "agreement_count", "average_exception_count"]
        ].copy()
        output["average_exception_count"] = pd.to_numeric(
            output["average_exception_count"], errors="coerce"
        ).round(2)
        tracker.record_semantic(
            "optimized_plan",
            len(documents),
            output,
            optimized.result,
            time.time() - started,
        )

        ordered = output.sort_values(
            ["agreement_count", "class_label"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(output), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
