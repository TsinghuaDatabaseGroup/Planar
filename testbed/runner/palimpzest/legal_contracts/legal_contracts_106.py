#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-106."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-106"
DATASET = "contract-exhibit"
PROFILE_COLUMNS = [
    "has_non_compete",
    "has_customer_non_solicitation",
    "has_employee_non_solicitation",
    "has_standstill",
    "has_vendor_non_solicitation",
]


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def normalize_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(str(value))
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
        values = parsed if isinstance(parsed, (list, tuple, set)) else [value]
    return [text for item in values if (text := normalize_text(item)) is not None]


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it has non-disclosure-agreement status, "
                "including a dual-status SEC exhibit."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        ndas = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), ndas, scope_result, time.time() - started
        )

        base_extraction_plan = memory_dataset(
            f"{TASK_ID}-base-extraction", ndas
        ).sem_map(
            cols=[
                {"name": "duration_years", "type": float | None,
                 "desc": "The expressly stated finite confidentiality duration normalized to years, or null when missing, perpetual, indefinite, or not quantifiable."},
                {"name": "governing_law_jurisdictions", "type": list[str],
                 "desc": "Every expressly stated governing-law jurisdiction; return an empty list if none is stated."},
            ],
            desc="Extract the finite confidentiality duration and governing-law jurisdictions.",
            depends_on=["text"],
        )
        started = time.time()
        base_extraction_result = base_extraction_plan.run(config)
        base_records = result_frame(
            base_extraction_result,
            ndas,
            ["duration_years", "governing_law_jurisdictions"],
        )
        base_records["duration_years"] = base_records["duration_years"].map(
            normalize_number
        )
        base_records["governing_law_jurisdictions"] = base_records[
            "governing_law_jurisdictions"
        ].map(normalize_string_list)
        tracker.record_semantic(
            "sem_map", len(ndas), base_records,
            base_extraction_result, time.time() - started,
        )

        finite_single_law = base_records.loc[
            base_records["duration_years"].notna()
            & base_records["governing_law_jurisdictions"].map(len).eq(1)
        ].reset_index(drop=True)
        tracker.record("filter", len(base_records), finite_single_law)

        profile_plan = memory_dataset(
            f"{TASK_ID}-profile", finite_single_law
        ).sem_map(
            cols=[
                {"name": "has_non_compete", "type": bool,
                 "desc": "True only if an operative non-compete clause is present; otherwise false."},
                {"name": "has_customer_non_solicitation", "type": bool,
                 "desc": "True only if an operative customer non-solicitation clause is present; otherwise false."},
                {"name": "has_employee_non_solicitation", "type": bool,
                 "desc": "True only if an operative employee non-solicitation clause is present; otherwise false."},
                {"name": "has_standstill", "type": bool,
                 "desc": "True only if an operative standstill clause is present; otherwise false."},
                {"name": "has_vendor_non_solicitation", "type": bool,
                 "desc": "True only if an operative vendor non-solicitation clause is present; otherwise false."},
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
        profiled = result_frame(profile_result, finite_single_law, PROFILE_COLUMNS)
        for column in PROFILE_COLUMNS:
            profiled[column] = profiled[column].map(parse_bool)
        profiled["restriction_profile"] = [
            tuple(bool(row[column]) for column in PROFILE_COLUMNS)
            for _, row in profiled.iterrows()
        ]
        tracker.record_semantic(
            "sem_map", len(finite_single_law), profiled,
            profile_result, time.time() - started,
        )

        restrictive = profiled.loc[
            profiled["restriction_profile"].map(any)
        ].reset_index(drop=True)
        tracker.record("filter", len(profiled), restrictive)

        eligible = restrictive[
            ["governing_law_jurisdictions", "restriction_profile", "duration_years"]
        ].copy()
        eligible["jurisdiction"] = eligible["governing_law_jurisdictions"].map(
            lambda values: values[0]
        )
        eligible = eligible[
            ["jurisdiction", "restriction_profile", "duration_years"]
        ].reset_index(drop=True)
        tracker.record("project", len(restrictive), eligible)

        jurisdiction_counts = (
            eligible.groupby("jurisdiction", sort=False)
            .size()
            .rename("qualifying_nda_count")
            .reset_index()
        )
        tracker.record("groupby", len(eligible), jurisdiction_counts)

        ordered_jurisdictions = jurisdiction_counts.sort_values(
            ["qualifying_nda_count", "jurisdiction"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(jurisdiction_counts), ordered_jurisdictions)

        top_jurisdictions = ordered_jurisdictions.head(3).reset_index(drop=True)
        tracker.record("limit", len(ordered_jurisdictions), top_jurisdictions)

        profiles = (
            eligible.groupby(["jurisdiction", "restriction_profile"], sort=False)
            .agg(
                profile_nda_count=("duration_years", "size"),
                profile_avg_duration_years=("duration_years", "mean"),
            )
            .reset_index()
        )
        profiles["profile_avg_duration_years"] = profiles[
            "profile_avg_duration_years"
        ].round(2)
        tracker.record("groupby", len(eligible), profiles)

        joined = top_jurisdictions.merge(profiles, on="jurisdiction", how="inner")
        tracker.record(
            "join",
            {"left": len(top_jurisdictions), "right": len(profiles)},
            joined,
        )

        ordered_profiles = joined.sort_values(
            [
                "qualifying_nda_count",
                "jurisdiction",
                "profile_nda_count",
                "restriction_profile",
            ],
            ascending=[False, True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(joined), ordered_profiles)

        most_common = ordered_profiles.drop_duplicates(
            subset=["jurisdiction"], keep="first", ignore_index=True
        )
        tracker.record("dedup", len(ordered_profiles), most_common)

        projected = most_common[
            [
                "jurisdiction",
                "qualifying_nda_count",
                "restriction_profile",
                "profile_nda_count",
                "profile_avg_duration_years",
            ]
        ].rename(columns={"restriction_profile": "most_common_restriction_profile"})
        projected["most_common_restriction_profile"] = projected[
            "most_common_restriction_profile"
        ].map(list)
        projected = projected.reset_index(drop=True)
        tracker.record("project", len(most_common), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
