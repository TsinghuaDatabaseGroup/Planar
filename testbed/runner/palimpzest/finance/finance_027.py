#!/usr/bin/env python3
"""Palimpzest pipeline for finance-027."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "finance-027"
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
    "undetermined",
)
COMPETITIVE_POSITIONS = (
    "emerging",
    "niche_player",
    "challenger",
    "major_player",
    "market_leader",
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
            "CSV/2021.csv",
            ["id", "cik", "text", "word_count"],
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[
            filings["word_count"] > 60_000
        ].reset_index(drop=True)
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["id", "cik", "word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        ai_plan = memory_dataset(f"{TASK_ID}-ai", documents).sem_filter(
            filter=(
                "Keep the filing only if it mentions artificial intelligence or "
                "machine learning."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        ai_result = ai_plan.run(config)
        ai_filings = result_frame(ai_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(ai_filings),
            ai_result,
            time.time() - started,
        )

        revenue_plan = memory_dataset(
            f"{TASK_ID}-revenue", ai_filings
        ).sem_map(
            cols=[
                {
                    "name": "revenue_model",
                    "type": str,
                    "desc": (
                        "Exactly one primary category: advertising, commission, "
                        "interest_income, investment_returns, licensing_royalties, "
                        "mixed, product_sales, rental_income, service_fees, "
                        "subscription, or undetermined when no primary category "
                        "can be established."
                    ),
                }
            ],
            desc="Assign the company's primary revenue-model category.",
            depends_on=["document_text"],
        )
        started = time.time()
        revenue_result = revenue_plan.run(config)
        revenue_labeled = result_frame(
            revenue_result,
            ai_filings,
            ["revenue_model"],
        )
        revenue_labeled["revenue_model"] = revenue_labeled[
            "revenue_model"
        ].map(lambda value: normalize_enum(value, REVENUE_MODELS))
        tracker.record_semantic(
            "sem_map",
            len(ai_filings),
            _intermediate_view(revenue_labeled),
            revenue_result,
            time.time() - started,
        )

        position_plan = memory_dataset(
            f"{TASK_ID}-position", revenue_labeled
        ).sem_map(
            cols=[
                {
                    "name": "competitive_position",
                    "type": str,
                    "desc": (
                        "Exactly one category: emerging, niche_player, challenger, "
                        "major_player, market_leader, or undetermined when the "
                        "competitive position cannot be established."
                    ),
                }
            ],
            desc="Assign the company's competitive-position category.",
            depends_on=["document_text"],
        )
        started = time.time()
        position_result = position_plan.run(config)
        position_labeled = result_frame(
            position_result,
            revenue_labeled,
            ["competitive_position"],
        )
        position_labeled["competitive_position"] = position_labeled[
            "competitive_position"
        ].map(lambda value: normalize_enum(value, COMPETITIVE_POSITIONS))
        tracker.record_semantic(
            "sem_map",
            len(revenue_labeled),
            _intermediate_view(position_labeled),
            position_result,
            time.time() - started,
        )

        determined = position_labeled.loc[
            position_labeled["revenue_model"].notna()
            & position_labeled["competitive_position"].notna()
            & position_labeled["revenue_model"].ne("undetermined")
            & position_labeled["competitive_position"].ne("undetermined")
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(position_labeled),
            _intermediate_view(determined),
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-summaries", determined
        ).sem_map(
            cols=[
                {
                    "name": "esg_mentioned",
                    "type": bool,
                    "desc": (
                        "True when the filing mentions environmental, social, or "
                        "governance topics or sustainability; otherwise false."
                    ),
                },
                {
                    "name": "risk_factor_count",
                    "type": int,
                    "desc": (
                        "The total number of risk factors disclosed in the filing's "
                        "Risk Factors section as an integer, or null when it cannot "
                        "be determined reliably."
                    ),
                },
            ],
            desc=(
                "Extract whether ESG or sustainability topics are mentioned and "
                "the total number of risk factors."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            determined,
            ["esg_mentioned", "risk_factor_count"],
        )
        extracted["esg_mentioned"] = extracted["esg_mentioned"].map(parse_bool)
        extracted["risk_factor_count"] = pd.to_numeric(
            extracted["risk_factor_count"], errors="coerce"
        )
        tracker.record_semantic(
            "sem_map",
            len(determined),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby(["revenue_model", "competitive_position"])
            .agg(
                filing_count=("id", "size"),
                esg_mention_rate_percent=("esg_mentioned", "mean"),
                median_risk_factor_count=("risk_factor_count", "median"),
            )
            .reset_index()
        )
        grouped["esg_mention_rate_percent"] = (
            100.0 * grouped["esg_mention_rate_percent"]
        ).round(1)
        grouped["median_risk_factor_count"] = grouped[
            "median_risk_factor_count"
        ].round(1)
        tracker.record("groupby", len(extracted), grouped)

        qualifying = grouped.loc[
            grouped["filing_count"] >= 10
        ].reset_index(drop=True)
        tracker.record("filter", len(grouped), qualifying)

        ordered = qualifying.sort_values(
            ["filing_count", "revenue_model", "competitive_position"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(qualifying), ordered)

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
