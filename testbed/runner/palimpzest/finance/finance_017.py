#!/usr/bin/env python3
"""Palimpzest pipeline for finance-017."""

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

TASK_ID = "finance-017"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2022.csv",
            ["id", "cik", "text", "word_count"],
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["id", "cik", "word_count", "document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        domestic_plan = memory_dataset(
            f"{TASK_ID}-domestic", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it does not describe substantive business "
                "operations outside the United States."
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

        esg_plan = memory_dataset(f"{TASK_ID}-esg", domestic).sem_filter(
            filter=(
                "Keep the filing only if it discusses environmental, social, or "
                "governance topics."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        esg_result = esg_plan.run(config)
        esg_filings = result_frame(esg_result, domestic)
        tracker.record_semantic(
            "sem_filter",
            len(domestic),
            _intermediate_view(esg_filings),
            esg_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-company", esg_filings
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The normalized legal name of the filing company.",
                },
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": "The company's primary industry sector.",
                },
            ],
            desc="Extract and normalize the company legal name and industry sector.",
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            esg_filings,
            ["company_name", "industry_sector"],
        )
        tracker.record_semantic(
            "sem_map",
            len(esg_filings),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        deduped = (
            extracted.assign(
                _company_key=extracted["company_name"].astype(str).str.casefold()
            )
            .sort_values(
                ["_company_key", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("_company_key", keep="first")
            .drop(columns=["_company_key"])
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(extracted), _intermediate_view(deduped))

        ordered = deduped.assign(
            _company_sort=deduped["company_name"].astype(str).str.casefold()
        ).sort_values(
            ["word_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        )
        tracker.record("order_by", len(deduped), _intermediate_view(ordered))

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), _intermediate_view(limited))

        answer_frame = limited[
            ["company_name", "word_count", "industry_sector"]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
