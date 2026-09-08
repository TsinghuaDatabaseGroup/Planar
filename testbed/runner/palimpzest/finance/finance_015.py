#!/usr/bin/env python3
"""Palimpzest pipeline for finance-015."""

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
    result_frame,
    save_output,
)

TASK_ID = "finance-015"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2024.csv",
            ["id", "cik", "text", "word_count"],
        )
        filings["word_count"] = filings["word_count"].astype(int)
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

        ai_plan = memory_dataset(f"{TASK_ID}-ai", documents).sem_filter(
            filter=(
                "Keep the filing only if it describes the company's operational "
                "use of artificial intelligence or machine learning."
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

        continuing_plan = memory_dataset(
            f"{TASK_ID}-continuing", ai_filings
        ).sem_filter(
            filter=(
                "Keep the filing only if it does not report substantial doubt "
                "about the company's ability to continue as a going concern."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        continuing_result = continuing_plan.run(config)
        continuing = result_frame(continuing_result, ai_filings)
        tracker.record_semantic(
            "sem_filter",
            len(ai_filings),
            _intermediate_view(continuing),
            continuing_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-company", continuing
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The company's legal name as stated in the filing.",
                },
                {
                    "name": "industry",
                    "type": str,
                    "desc": "The company's primary industry sector.",
                },
            ],
            desc="Extract the company legal name and primary industry sector.",
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            continuing,
            ["company_name", "industry"],
        )
        tracker.record_semantic(
            "sem_map",
            len(continuing),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        ordered = extracted.assign(
            _company_sort=extracted["company_name"].astype(str).str.casefold()
        ).sort_values(
            ["word_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        )
        tracker.record("order_by", len(extracted), _intermediate_view(ordered))

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), _intermediate_view(limited))

        answer = df_records(
            limited[["company_name", "industry", "word_count"]]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
