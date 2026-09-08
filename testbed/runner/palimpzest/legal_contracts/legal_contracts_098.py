#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-098."""

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

TASK_ID = "legal_contracts-098"
DATASET = "contract-exhibit"
PROFILE_COLUMNS = [
    "has_non_compete",
    "has_customer_non_solicitation",
    "has_employee_non_solicitation",
    "has_standstill",
    "has_vendor_non_solicitation",
]


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it has non-disclosure-agreement status, "
                "including documents dual-labelled as an NDA and another SEC exhibit type."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        duration_plan = memory_dataset(
            f"{TASK_ID}-duration", scoped
        ).sem_map(
            cols=[
                {"name": "duration_years", "type": float | None,
                 "desc": "The expressly stated finite confidentiality duration normalized to years, or null when the term is missing, perpetual, indefinite, or not quantifiable."}
            ],
            desc="Extract the finite confidentiality duration in years.",
            depends_on=["text"],
        )
        started = time.time()
        duration_result = duration_plan.run(config)
        durations = result_frame(duration_result, scoped, ["duration_years"])
        durations["duration_years"] = durations["duration_years"].map(
            normalize_number
        )
        tracker.record_semantic(
            "sem_map", len(scoped), durations, duration_result, time.time() - started
        )

        finite_durations = durations.loc[
            durations["duration_years"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(durations), finite_durations)

        profile_plan = memory_dataset(
            f"{TASK_ID}-profile", finite_durations
        ).sem_map(
            cols=[
                {"name": "has_non_compete", "type": bool,
                 "desc": "True only if the NDA contains an operative non-compete clause; otherwise false."},
                {"name": "has_customer_non_solicitation", "type": bool,
                 "desc": "True only if the NDA contains an operative customer non-solicitation clause; otherwise false."},
                {"name": "has_employee_non_solicitation", "type": bool,
                 "desc": "True only if the NDA contains an operative employee non-solicitation clause; otherwise false."},
                {"name": "has_standstill", "type": bool,
                 "desc": "True only if the NDA contains an operative standstill clause; otherwise false."},
                {"name": "has_vendor_non_solicitation", "type": bool,
                 "desc": "True only if the NDA contains an operative vendor non-solicitation clause; otherwise false."},
            ],
            desc=(
                "Determine the exact five-position boolean restriction profile in "
                "this order: non-compete, customer non-solicitation, employee "
                "non-solicitation, standstill, vendor non-solicitation."
            ),
            depends_on=["text"],
        )
        started = time.time()
        profile_result = profile_plan.run(config)
        profiled = result_frame(profile_result, finite_durations, PROFILE_COLUMNS)
        for column in PROFILE_COLUMNS:
            profiled[column] = profiled[column].map(parse_bool)
        profiled["restriction_profile"] = [
            tuple(bool(row[column]) for column in PROFILE_COLUMNS)
            for _, row in profiled.iterrows()
        ]
        profiled = profiled[
            ["restriction_profile", "duration_years"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(finite_durations), profiled,
            profile_result, time.time() - started,
        )

        grouped = (
            profiled.groupby("restriction_profile", sort=False)
            .agg(
                member_count=("duration_years", "size"),
                avg_duration_years=("duration_years", "mean"),
            )
            .reset_index()
        )
        grouped["avg_duration_years"] = grouped["avg_duration_years"].round(2)
        grouped["restriction_profile"] = grouped["restriction_profile"].map(list)
        tracker.record("groupby", len(profiled), grouped)

        ordered = grouped.sort_values(
            "member_count", ascending=False, kind="stable"
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
