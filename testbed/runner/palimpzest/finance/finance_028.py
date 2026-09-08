#!/usr/bin/env python3
"""Palimpzest pipeline for finance-028."""

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

TASK_ID = "finance-028"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2021.csv",
            ["id", "cik", "text"],
        )
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["id", "cik", "document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        concern_plan = memory_dataset(
            f"{TASK_ID}-going-concern", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it reports substantial doubt about the "
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
                "Keep the filing only if it reports pending litigation involving "
                "the company."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        litigation_result = litigation_plan.run(config)
        litigation_filings = result_frame(litigation_result, concern_filings)
        tracker.record_semantic(
            "sem_filter",
            len(concern_filings),
            _intermediate_view(litigation_filings),
            litigation_result,
            time.time() - started,
        )

        customer_plan = memory_dataset(
            f"{TASK_ID}-customer-concentration", litigation_filings
        ).sem_filter(
            filter=(
                "Keep the filing only if it reports that a single customer accounts "
                "for at least 10 percent of revenue."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        customer_result = customer_plan.run(config)
        concentrated_filings = result_frame(customer_result, litigation_filings)
        tracker.record_semantic(
            "sem_filter",
            len(litigation_filings),
            _intermediate_view(concentrated_filings),
            customer_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-industry", concentrated_filings
        ).sem_map(
            cols=[
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": "The company's primary industry sector.",
                }
            ],
            desc="Extract the industry sector.",
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            concentrated_filings,
            ["industry_sector"],
        )
        tracker.record_semantic(
            "sem_map",
            len(concentrated_filings),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        deduped = extracted.drop_duplicates("cik", keep="first").reset_index(
            drop=True
        )
        tracker.record("dedup", len(extracted), _intermediate_view(deduped))

        grouped = (
            deduped.groupby("industry_sector")
            .size()
            .rename("company_count")
            .reset_index()
        )
        tracker.record("groupby", len(deduped), grouped)

        ordered = grouped.sort_values(
            ["company_count", "industry_sector"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(grouped), ordered)

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
