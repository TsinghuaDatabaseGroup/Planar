#!/usr/bin/env python3
"""Palimpzest pipeline for finance-004."""

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

TASK_ID = "finance-004"
DATASET = "SEC"
REVENUE_MODELS = (
    "mixed",
    "product_sales",
    "interest_income",
    "service_fees",
    "licensing_royalties",
    "rental_income",
    "investment_returns",
    "subscription",
    "commission",
)


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2023.csv",
            ["text", "word_count"],
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[filings["word_count"] > 100_000].reset_index(
            drop=True
        )
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        classification_plan = memory_dataset(
            f"{TASK_ID}-revenue-model", documents
        ).sem_map(
            cols=[
                {
                    "name": "revenue_model",
                    "type": str,
                    "desc": (
                        "Exactly one primary category: mixed, product_sales, "
                        "interest_income, service_fees, licensing_royalties, "
                        "rental_income, investment_returns, subscription, or "
                        "commission. Leave unassigned when no primary model can be "
                        "determined."
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

        grouped = (
            determined.groupby("revenue_model", sort=False)
            .size()
            .rename("filing_count")
            .reset_index()
        )
        tracker.record("groupby", len(determined), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
