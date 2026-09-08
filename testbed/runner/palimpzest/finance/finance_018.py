#!/usr/bin/env python3
"""Palimpzest pipeline for finance-018."""

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

TASK_ID = "finance-018"
DATASET = "SEC"
REVENUE_MODELS = (
    "advertising",
    "commission",
    "interest_income",
    "investment_returns",
    "licensing_royalties",
    "mixed",
    "product_sales",
    "rental_income",
    "service_fees",
    "subscription",
)


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

        classification_plan = memory_dataset(
            f"{TASK_ID}-revenue-model", documents
        ).sem_map(
            cols=[
                {
                    "name": "revenue_model",
                    "type": str,
                    "desc": (
                        "Exactly one primary category: advertising, commission, "
                        "interest_income, investment_returns, licensing_royalties, "
                        "mixed, product_sales, rental_income, service_fees, or "
                        "subscription. Leave unassigned when no primary model can "
                        "be determined."
                    ),
                }
            ],
            desc=(
                "Assign the filing to exactly one primary revenue-model category "
                "according to how the reporting company primarily earns revenue."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            documents,
            ["revenue_model"],
        )
        classified["revenue_model"] = classified["revenue_model"].map(
            lambda value: normalize_enum(value, REVENUE_MODELS)
        )
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(classified),
            classification_result,
            time.time() - started,
        )

        determined = classified.loc[
            classified["revenue_model"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), _intermediate_view(determined))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-employees", determined
        ).sem_map(
            cols=[
                {
                    "name": "employee_count",
                    "type": int,
                    "desc": (
                        "The explicitly reported company-wide total employee count, "
                        "or null when no explicit total is disclosed."
                    ),
                }
            ],
            desc="Extract the reported employee count.",
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            determined,
            ["employee_count"],
        )
        extracted["employee_count"] = pd.to_numeric(
            extracted["employee_count"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(determined),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("revenue_model", sort=False)
            .agg(
                filing_count=("revenue_model", "size"),
                average_employee_count=("employee_count", "mean"),
            )
            .reset_index()
        )
        grouped["average_employee_count"] = pd.to_numeric(
            grouped["average_employee_count"], errors="coerce"
        ).round(0)
        tracker.record("groupby", len(extracted), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
