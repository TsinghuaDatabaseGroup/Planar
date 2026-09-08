#!/usr/bin/env python3
"""LOTUS pipeline for finance-027."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    iter_selected_texts,
    load_table,
    normalize_enum,
    parse_bool,
    parse_number,
    save_output,
    sem_extract_in_batches,
    sem_map_in_batches,
    setup,
)

TASK_ID = "finance-027"
BATCH_SIZE = 100
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


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _intermediate_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["document_text"], errors="ignore")


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2021.csv")[
            ["id", "cik", "text", "word_count"]
        ].copy()
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2021.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 60000].copy()
        tracker.record(
            "FILTER(word_count > 60000)",
            len(filings),
            len(candidates),
            output=candidates,
        )
        tracker.record(
            "SCAN_DOCS(selector=filtered.text)",
            len(candidates),
            len(candidates),
            output=candidates,
        )

        with tracker.step(
            "SEM_FILTER(mentions artificial intelligence or machine learning)",
            input_rows=len(candidates),
        ) as step:
            ai_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                ai_parts.append(
                    documents.sem_filter(
                        "The 10-K filing {document_text} mentions artificial "
                        "intelligence or machine learning."
                    )
                )
            ai_filings = (
                pd.concat(ai_parts, ignore_index=True)
                if ai_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(ai_filings))

        with tracker.step(
            "SEM_CLASSIFY(primary revenue-model category)",
            input_rows=len(ai_filings),
        ) as step:
            revenue_labeled = sem_map_in_batches(
                ai_filings,
                "Classify the company's primary revenue model from the 10-K filing "
                "{document_text}. Output exactly one label: advertising, commission, "
                "interest_income, investment_returns, licensing_royalties, mixed, "
                "product_sales, rental_income, service_fees, subscription, or "
                "undetermined.",
                suffix="revenue_model",
                batch_size=BATCH_SIZE,
            )
            revenue_labeled["revenue_model"] = revenue_labeled["revenue_model"].map(
                lambda value: normalize_enum(value, REVENUE_MODELS)
            )
            step.set_output(_intermediate_view(revenue_labeled))

        with tracker.step(
            "SEM_CLASSIFY(competitive-position category)",
            input_rows=len(revenue_labeled),
        ) as step:
            position_labeled = sem_map_in_batches(
                revenue_labeled,
                "Classify the company's competitive position from the 10-K filing "
                "{document_text}. Output exactly one label: emerging, niche_player, "
                "challenger, major_player, market_leader, or undetermined.",
                suffix="competitive_position",
                batch_size=BATCH_SIZE,
            )
            position_labeled["competitive_position"] = position_labeled[
                "competitive_position"
            ].map(lambda value: normalize_enum(value, COMPETITIVE_POSITIONS))
            step.set_output(_intermediate_view(position_labeled))

        determined = position_labeled[
            position_labeled["revenue_model"].notna()
            & position_labeled["competitive_position"].notna()
            & (position_labeled["revenue_model"] != "undetermined")
            & (position_labeled["competitive_position"] != "undetermined")
        ].copy()
        tracker.record(
            "FILTER(revenue_model != 'undetermined' AND competitive_position != 'undetermined')",
            len(position_labeled),
            len(determined),
            output=_intermediate_view(determined),
        )

        with tracker.step(
            "SEM_EXTRACT(ESG mention and total risk-factor count)",
            input_rows=len(determined),
        ) as step:
            extracted = sem_extract_in_batches(
                determined,
                input_cols=["document_text"],
                output_cols={
                    "esg_mentioned": (
                        "true when the filing mentions environmental, social, or "
                        "governance topics or sustainability; false otherwise"
                    ),
                    "risk_factor_count": (
                        "the total number of risk factors disclosed in the filing's "
                        "Risk Factors section as an integer, or null when it cannot "
                        "be determined reliably"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted["esg_mentioned"] = extracted["esg_mentioned"].map(parse_bool)
            extracted["risk_factor_count"] = pd.to_numeric(
                extracted["risk_factor_count"].map(parse_number), errors="coerce"
            )
            step.set_output(_intermediate_view(extracted))

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
            100.0
            * pd.to_numeric(
                grouped["esg_mention_rate_percent"], errors="coerce"
            )
        ).round(1)
        grouped["median_risk_factor_count"] = pd.to_numeric(
            grouped["median_risk_factor_count"], errors="coerce"
        ).round(1)
        tracker.record(
            "GROUP_BY(revenue_model, competitive_position, COUNT, ESG_RATE, MEDIAN(risk_factor_count))",
            len(extracted),
            len(grouped),
            output=grouped,
        )

        qualifying = grouped[grouped["filing_count"] >= 10].copy()
        tracker.record(
            "FILTER(filing_count >= 10)",
            len(grouped),
            len(qualifying),
            output=qualifying,
        )

        answer_frame = qualifying.sort_values(
            ["filing_count", "revenue_model", "competitive_position"],
            ascending=[False, True, True],
            kind="mergesort",
        ).head(5).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(filing_count DESC, revenue_model ASC, competitive_position ASC) -> LIMIT(5)",
            len(qualifying),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
