#!/usr/bin/env python3
"""Palimpzest pipeline for finance-025."""

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
    result_frame,
    save_output,
)

TASK_ID = "finance-025"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2019.csv",
            ["id", "cik", "text", "word_count"],
        )
        filings["cik"] = pd.to_numeric(
            filings["cik"], errors="coerce"
        ).astype("Int64")
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[
            filings["word_count"] > 45_000
        ].reset_index(drop=True)
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["id", "cik", "word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        customer_plan = memory_dataset(
            f"{TASK_ID}-customer", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it discloses dependence on a major "
                "customer or material customer concentration."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        customer_result = customer_plan.run(config)
        customer_filings = result_frame(customer_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(customer_filings),
            customer_result,
            time.time() - started,
        )

        currency_plan = memory_dataset(
            f"{TASK_ID}-currency", customer_filings
        ).sem_filter(
            filter=(
                "Keep the filing only if it does not disclose foreign-currency "
                "risk or exchange-rate risk."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        currency_result = currency_plan.run(config)
        no_currency_risk = result_frame(currency_result, customer_filings)
        tracker.record_semantic(
            "sem_filter",
            len(customer_filings),
            _intermediate_view(no_currency_risk),
            currency_result,
            time.time() - started,
        )

        international_plan = memory_dataset(
            f"{TASK_ID}-international", no_currency_risk
        ).sem_filter(
            filter=(
                "Keep the filing only if it describes company operations outside "
                "the United States."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        international_result = international_plan.run(config)
        international = result_frame(international_result, no_currency_risk)
        tracker.record_semantic(
            "sem_filter",
            len(no_currency_risk),
            _intermediate_view(international),
            international_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-company", international
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The company's legal name as stated in the filing.",
                },
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": "The company's primary industry sector.",
                },
                {
                    "name": "employee_count",
                    "type": int,
                    "desc": (
                        "The total reported employee count as an integer, or null "
                        "when no determinable total is reported."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name, primary industry sector, and "
                "reported employee count."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            international,
            ["company_name", "industry_sector", "employee_count"],
        )
        extracted["employee_count"] = pd.to_numeric(
            extracted["employee_count"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(international),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        valid = extracted.loc[
            extracted["employee_count"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), _intermediate_view(valid))

        deduped = (
            valid.assign(
                _company_key=valid["company_name"].astype(str).str.casefold()
            )
            .sort_values(
                ["_company_key", "employee_count", "word_count", "cik"],
                ascending=[True, False, False, False],
                na_position="last",
                kind="mergesort",
            )
            .drop_duplicates("_company_key", keep="first")
            .drop(columns=["_company_key"])
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(valid), _intermediate_view(deduped))

        ordered = deduped.assign(
            _company_sort=deduped["company_name"].astype(str).str.casefold()
        ).sort_values(
            ["employee_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        )
        tracker.record("order_by", len(deduped), _intermediate_view(ordered))

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), _intermediate_view(limited))

        answer_frame = limited[
            ["company_name", "industry_sector", "employee_count"]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
