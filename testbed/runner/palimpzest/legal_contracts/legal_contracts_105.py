#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-105."""

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

TASK_ID = "legal_contracts-105"
DATASET = "contract-exhibit"
CORE_COLUMNS = [
    "has_confidential_information_definition",
    "has_nondisclosure",
    "has_non_use",
    "has_return_of_materials",
    "has_survival",
]
EXCEPTION_COLUMNS = [
    "has_public_information_exception",
    "has_prior_knowledge_exception",
    "has_independent_development_exception",
    "has_third_party_receipt_exception",
    "has_legal_compulsion_exception",
    "has_consent_exception",
]


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_iso_date(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


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

        date_law_plan = memory_dataset(
            f"{TASK_ID}-date-law", ndas
        ).sem_map(
            cols=[
                {"name": "effective_date", "type": str | None,
                 "desc": "The explicit agreement effective date in YYYY-MM-DD form, or null if none is stated."},
                {"name": "governing_law_jurisdictions", "type": list[str],
                 "desc": "Every expressly stated governing-law jurisdiction; return an empty list if none is stated."},
            ],
            desc="Extract the explicit effective date and governing-law jurisdictions.",
            depends_on=["text"],
        )
        started = time.time()
        date_law_result = date_law_plan.run(config)
        dated = result_frame(
            date_law_result,
            ndas,
            ["effective_date", "governing_law_jurisdictions"],
        )
        dated["effective_date"] = dated["effective_date"].map(normalize_iso_date)
        dated["governing_law_jurisdictions"] = dated[
            "governing_law_jurisdictions"
        ].map(normalize_string_list)
        tracker.record_semantic(
            "sem_map", len(ndas), dated, date_law_result, time.time() - started
        )

        parsed_dates = pd.to_datetime(dated["effective_date"], errors="coerce")
        eligible = dated.loc[
            parsed_dates.ge(pd.Timestamp("2010-01-01"))
            & dated["governing_law_jurisdictions"].map(len).eq(1)
        ].reset_index(drop=True)
        tracker.record("filter", len(dated), eligible)

        feature_plan = memory_dataset(
            f"{TASK_ID}-features", eligible
        ).sem_map(
            cols=[
                {"name": "has_confidential_information_definition", "type": bool,
                 "desc": "True only if confidential information is explicitly defined; otherwise false."},
                {"name": "has_nondisclosure", "type": bool,
                 "desc": "True only if an operative nondisclosure obligation is present; otherwise false."},
                {"name": "has_non_use", "type": bool,
                 "desc": "True only if an operative non-use obligation is present; otherwise false."},
                {"name": "has_return_of_materials", "type": bool,
                 "desc": "True only if an operative return-of-materials obligation is present; otherwise false."},
                {"name": "has_survival", "type": bool,
                 "desc": "True only if an operative survival clause is present; otherwise false."},
                {"name": "has_public_information_exception", "type": bool,
                 "desc": "True only if a public-information exception is present; otherwise false."},
                {"name": "has_prior_knowledge_exception", "type": bool,
                 "desc": "True only if a prior-knowledge exception is present; otherwise false."},
                {"name": "has_independent_development_exception", "type": bool,
                 "desc": "True only if an independent-development exception is present; otherwise false."},
                {"name": "has_third_party_receipt_exception", "type": bool,
                 "desc": "True only if a third-party-receipt exception is present; otherwise false."},
                {"name": "has_legal_compulsion_exception", "type": bool,
                 "desc": "True only if a legal-compulsion exception is present; otherwise false."},
                {"name": "has_consent_exception", "type": bool,
                 "desc": "True only if a consent exception is present; otherwise false."},
                {"name": "perpetual_confidentiality_term", "type": bool,
                 "desc": "True only if the confidentiality term is perpetual; otherwise false."},
            ],
            desc=(
                "Determine the five named core clauses, six standard exceptions, "
                "and whether the confidentiality term is perpetual."
            ),
            depends_on=["text"],
        )
        started = time.time()
        feature_result = feature_plan.run(config)
        feature_columns = [
            *CORE_COLUMNS,
            *EXCEPTION_COLUMNS,
            "perpetual_confidentiality_term",
        ]
        features = result_frame(feature_result, eligible, feature_columns)
        for column in feature_columns:
            features[column] = features[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(eligible), features, feature_result, time.time() - started
        )

        projected = pd.DataFrame(
            {
                "jurisdiction": features["governing_law_jurisdictions"].map(
                    lambda values: values[0]
                ),
                "all_five_core_clauses": features[CORE_COLUMNS].all(axis=1),
                "all_six_standard_exceptions": features[EXCEPTION_COLUMNS].all(axis=1),
                "perpetual_confidentiality_term": features[
                    "perpetual_confidentiality_term"
                ],
            }
        )
        tracker.record("project", len(features), projected)

        grouped = (
            projected.groupby("jurisdiction", sort=False)
            .agg(
                nda_count=("jurisdiction", "size"),
                all_five_core_clauses_count=("all_five_core_clauses", "sum"),
                all_six_standard_exceptions_count=(
                    "all_six_standard_exceptions",
                    "sum",
                ),
                perpetual_term_count=("perpetual_confidentiality_term", "sum"),
            )
            .reset_index()
        )
        for column in (
            "nda_count",
            "all_five_core_clauses_count",
            "all_six_standard_exceptions_count",
            "perpetual_term_count",
        ):
            grouped[column] = grouped[column].astype(int)
        tracker.record("groupby", len(projected), grouped)

        ordered = grouped.sort_values(
            "nda_count", ascending=False, kind="stable"
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)

        limited = ordered.head(3).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
