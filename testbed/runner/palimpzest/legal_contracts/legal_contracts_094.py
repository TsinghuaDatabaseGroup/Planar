#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-094."""

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

TASK_ID = "legal_contracts-094"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_direction(value) -> str | None:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().casefold()
    ).strip("_")
    return normalized if normalized in {"mutual", "unilateral"} else None


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
                "Keep the document only if it is a non-disclosure agreement, "
                "including an exhibit dual-labelled as an NDA and another agreement type."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        feature_plan = memory_dataset(
            f"{TASK_ID}-features", scoped
        ).sem_filter(
            filter=(
                "Keep the agreement only if it explicitly defines confidential "
                "information and contains both an operative return-or-destroy "
                "obligation and injunctive or equitable relief language."
            ),
            depends_on=["text"],
        )
        started = time.time()
        feature_result = feature_plan.run(config)
        qualifying = result_frame(feature_result, scoped)
        tracker.record_semantic(
            "sem_filter", len(scoped), qualifying, feature_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", qualifying
        ).sem_map(
            cols=[
                {"name": "governing_law", "type": str | None,
                 "desc": "The normalized expressly stated governing-law jurisdiction, using a consistent full jurisdiction name, or null when none is stated."},
                {"name": "nda_direction", "type": str | None,
                 "desc": "Mutual if both parties owe confidentiality duties, unilateral if only one party does, otherwise null."},
                {"name": "finite_term_years", "type": float | None,
                 "desc": "The expressly stated finite agreement term normalized to years, or null for perpetual, unspecified, or unquantifiable terms."},
            ],
            desc="Extract governing law, NDA direction, and finite agreement term.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            qualifying,
            ["governing_law", "nda_direction", "finite_term_years"],
        )
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        extracted["nda_direction"] = extracted["nda_direction"].map(
            normalize_direction
        )
        extracted["finite_term_years"] = extracted["finite_term_years"].map(
            normalize_number
        )
        extracted = extracted[
            ["governing_law", "nda_direction", "finite_term_years"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(qualifying), extracted, extraction_result,
            time.time() - started,
        )

        filtered = extracted.loc[
            extracted["governing_law"].notna()
            & extracted["nda_direction"].isin(["mutual", "unilateral"])
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), filtered)

        grouped = (
            filtered.groupby("governing_law", sort=False)
            .agg(
                mutual_count=(
                    "nda_direction",
                    lambda values: int(values.eq("mutual").sum()),
                ),
                unilateral_count=(
                    "nda_direction",
                    lambda values: int(values.eq("unilateral").sum()),
                ),
                average_finite_term_years=("finite_term_years", "mean"),
            )
            .reset_index()
        )
        grouped["average_finite_term_years"] = grouped[
            "average_finite_term_years"
        ].round(2)
        tracker.record("groupby", len(filtered), grouped)

        ordered = grouped.assign(
            combined_count=grouped["mutual_count"] + grouped["unilateral_count"]
        ).sort_values(
            ["combined_count", "governing_law"],
            ascending=[False, True],
            kind="stable",
        )
        ordered = ordered.drop(columns=["combined_count"]).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)

        limited = ordered.head(3).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
