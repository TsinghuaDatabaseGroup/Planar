#!/usr/bin/env python3
"""Palimpzest pipeline for finance-011."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "finance-011"
DATASET = "SEC"
GEOGRAPHIC_LEVELS = (
    "domestic_only",
    "limited_international",
    "globally_diversified",
    "unassigned",
)
ASSIGNED_LEVELS = GEOGRAPHIC_LEVELS[:3]


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(DATASET, "CSV/2023.csv", ["text"])
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-geography", documents
        ).sem_map(
            cols=[
                {
                    "name": "geographic_level",
                    "type": str,
                    "desc": (
                        "Exactly one operating-footprint category: domestic_only "
                        "when operations are confined to the United States, "
                        "limited_international when operations outside the United "
                        "States are geographically limited, globally_diversified "
                        "when operations span a broad set of countries or regions, "
                        "or unassigned when the scope cannot be determined."
                    ),
                },
                {
                    "name": "foreign_currency_risk",
                    "type": bool,
                    "desc": (
                        "True when the filing reports foreign-currency or "
                        "exchange-rate risk; false otherwise."
                    ),
                },
                {
                    "name": "has_international_ops",
                    "type": bool,
                    "desc": (
                        "True when the filing describes actual company operations "
                        "outside the United States; false otherwise."
                    ),
                },
                {
                    "name": "has_litigation",
                    "type": bool,
                    "desc": (
                        "True when the filing reports litigation, lawsuits, legal "
                        "proceedings, or regulatory investigations involving the "
                        "company; false otherwise."
                    ),
                },
            ],
            desc=(
                "Classify the geographic operating scope and determine whether the "
                "filing reports foreign-currency risk, international operations, "
                "and litigation or legal proceedings."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            [
                "geographic_level",
                "foreign_currency_risk",
                "has_international_ops",
                "has_litigation",
            ],
        )
        extracted["geographic_level"] = extracted["geographic_level"].map(
            lambda value: normalize_enum(value, GEOGRAPHIC_LEVELS)
        )
        for column in (
            "foreign_currency_risk",
            "has_international_ops",
            "has_litigation",
        ):
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        assigned = extracted.loc[
            extracted["geographic_level"].isin(ASSIGNED_LEVELS)
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), _intermediate_view(assigned))

        grouped = (
            assigned.groupby("geographic_level", sort=False)
            .agg(
                filing_count=("geographic_level", "size"),
                foreign_currency_risk_rate_percent=(
                    "foreign_currency_risk",
                    "mean",
                ),
                international_operations_rate_percent=(
                    "has_international_ops",
                    "mean",
                ),
                litigation_rate_percent=("has_litigation", "mean"),
            )
            .reset_index()
        )
        for column in (
            "foreign_currency_risk_rate_percent",
            "international_operations_rate_percent",
            "litigation_rate_percent",
        ):
            grouped[column] = (100.0 * grouped[column]).round(1)
        tracker.record("groupby", len(assigned), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
