#!/usr/bin/env python3
"""Palimpzest pipeline for finance-026."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "finance-026"
DATASET = "SEC"
GROWTH_STRATEGIES = (
    "organic",
    "acquisition_driven",
    "hybrid",
    "not_applicable",
    "undetermined",
)


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2020.csv",
            ["id", "cik", "text", "word_count"],
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[
            filings["word_count"] > 50_000
        ].reset_index(drop=True)
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["id", "cik", "word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        concern_plan = memory_dataset(
            f"{TASK_ID}-concern", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it expresses substantial doubt about the "
                "company's ability to continue as a going concern."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        concern_result = concern_plan.run(config)
        concern_filings = result_frame(concern_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(concern_filings),
            concern_result,
            time.time() - started,
        )

        litigation_plan = memory_dataset(
            f"{TASK_ID}-litigation", concern_filings
        ).sem_filter(
            filter=(
                "Keep the filing only if it reports pending, threatened, or "
                "ongoing litigation, legal proceedings, lawsuits, or regulatory "
                "investigations."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        litigation_result = litigation_plan.run(config)
        litigation = result_frame(litigation_result, concern_filings)
        tracker.record_semantic(
            "sem_filter",
            len(concern_filings),
            _intermediate_view(litigation),
            litigation_result,
            time.time() - started,
        )

        strategy_plan = memory_dataset(
            f"{TASK_ID}-strategy", litigation
        ).sem_map(
            cols=[
                {
                    "name": "growth_strategy",
                    "type": str,
                    "desc": (
                        "Exactly organic when the stated growth strategy is "
                        "primarily internal expansion; acquisition_driven when it "
                        "is primarily acquisitions; hybrid when both are stated "
                        "components; not_applicable when no growth strategy "
                        "applies; otherwise undetermined."
                    ),
                }
            ],
            desc="Assign the filing to its stated growth-strategy category.",
            depends_on=["document_text"],
        )
        started = time.time()
        strategy_result = strategy_plan.run(config)
        classified = result_frame(
            strategy_result,
            litigation,
            ["growth_strategy"],
        )
        classified["growth_strategy"] = classified["growth_strategy"].map(
            lambda value: normalize_enum(value, GROWTH_STRATEGIES)
        )
        tracker.record_semantic(
            "sem_map",
            len(litigation),
            _intermediate_view(classified),
            strategy_result,
            time.time() - started,
        )

        determined = classified.loc[
            classified["growth_strategy"].notna()
            & classified["growth_strategy"].ne("undetermined")
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), _intermediate_view(determined))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-counts", determined
        ).sem_map(
            cols=[
                {
                    "name": "employee_count",
                    "type": int,
                    "desc": (
                        "The total reported employee count as an integer, or null "
                        "when it cannot be determined."
                    ),
                },
                {
                    "name": "risk_factor_count",
                    "type": int,
                    "desc": (
                        "The total number of risk factors disclosed in the filing's "
                        "Risk Factors section as an integer, or null when it cannot "
                        "be determined."
                    ),
                },
            ],
            desc=(
                "Extract the reported employee count and total number of risk "
                "factors."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            determined,
            ["employee_count", "risk_factor_count"],
        )
        for column in ("employee_count", "risk_factor_count"):
            extracted[column] = pd.to_numeric(
                extracted[column], errors="coerce"
            ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(determined),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("growth_strategy", dropna=False)
            .agg(
                filing_count=("id", "size"),
                median_employee_count=("employee_count", "median"),
                median_risk_factor_count=("risk_factor_count", "median"),
            )
            .reset_index()
        )
        grouped["median_employee_count"] = grouped[
            "median_employee_count"
        ].round(1)
        grouped["median_risk_factor_count"] = grouped[
            "median_risk_factor_count"
        ].round(1)
        tracker.record("groupby", len(extracted), grouped)

        qualifying = grouped.loc[
            grouped["filing_count"] >= 10
        ].reset_index(drop=True)
        tracker.record("filter", len(grouped), qualifying)

        ordered = qualifying.sort_values(
            ["filing_count", "growth_strategy"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(qualifying), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
