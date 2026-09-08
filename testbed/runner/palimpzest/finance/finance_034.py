#!/usr/bin/env python3
"""Palimpzest pipeline for finance-034."""

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
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "finance-034"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def _clean_nullable_text(value) -> str | None:
    return normalize_text_value(value)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2023.csv",
            ["cik", "text", "word_count"],
        )
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[filings["word_count"] > 60_000].reset_index(
            drop=True
        )
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["cik", "word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        domestic_plan = memory_dataset(
            f"{TASK_ID}-domestic", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it describes the reporting company's "
                "operating footprint as exclusively domestic, with no operations "
                "outside the United States."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        domestic_result = domestic_plan.run(config)
        domestic = result_frame(domestic_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(domestic),
            domestic_result,
            time.time() - started,
        )

        currency_plan = memory_dataset(
            f"{TASK_ID}-currency-risk", domestic
        ).sem_filter(
            filter=(
                "Keep the filing only if it discloses foreign-currency or "
                "exchange-rate risk."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        currency_result = currency_plan.run(config)
        currency_risk = result_frame(currency_result, domestic)
        tracker.record_semantic(
            "sem_filter",
            len(domestic),
            _intermediate_view(currency_risk),
            currency_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-company-details", currency_risk
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": (
                        "The reporting company's legal name, or null when it "
                        "cannot be determined."
                    ),
                },
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": (
                        "The reporting company's industry sector, or null when it "
                        "cannot be determined."
                    ),
                },
                {
                    "name": "risk_factor_count",
                    "type": int,
                    "desc": (
                        "The total number of distinct risk factors in the filing's "
                        "Risk Factors section, or null when the total cannot be "
                        "determined."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name, industry sector, and total number "
                "of risk factors."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            currency_risk,
            ["company_name", "industry_sector", "risk_factor_count"],
        )
        extracted["company_name"] = extracted["company_name"].map(
            _clean_nullable_text
        )
        extracted["industry_sector"] = extracted["industry_sector"].map(
            _clean_nullable_text
        )
        extracted["risk_factor_count"] = pd.to_numeric(
            extracted["risk_factor_count"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(currency_risk),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        complete = extracted.loc[
            extracted["company_name"].notna()
            & extracted["industry_sector"].notna()
            & extracted["risk_factor_count"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), _intermediate_view(complete))

        deduped = (
            complete.sort_values(
                ["company_name", "risk_factor_count", "word_count", "cik"],
                ascending=[True, False, False, False],
                kind="mergesort",
            )
            .drop_duplicates("company_name", keep="first")
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(complete), _intermediate_view(deduped))

        ordered = deduped.assign(
            _company_sort=deduped["company_name"].str.casefold()
        ).sort_values(
            ["risk_factor_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        )
        tracker.record("order_by", len(deduped), _intermediate_view(ordered))

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), _intermediate_view(limited))

        answer_frame = limited[
            ["company_name", "industry_sector", "risk_factor_count"]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
