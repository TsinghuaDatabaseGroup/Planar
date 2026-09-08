#!/usr/bin/env python3
"""Palimpzest pipeline for finance-001."""

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

TASK_ID = "finance-001"
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
            ["id", "cik", "text"],
        )
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["id", "cik", "document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        retail_plan = memory_dataset(
            f"{TASK_ID}-retail", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it describes a company operating in a "
                "retail industry."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        retail_result = retail_plan.run(config)
        retail = result_frame(retail_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(retail),
            retail_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-company", retail
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The company's legal name as stated in the filing.",
                },
                {
                    "name": "employee_count",
                    "type": int,
                    "desc": "The total reported employee count as an integer.",
                },
                {
                    "name": "retail_subsector",
                    "type": str,
                    "desc": "The company's specific retail subsector.",
                },
            ],
            desc=(
                "Extract the company legal name, reported employee count, and "
                "specific retail subsector."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            retail,
            ["company_name", "employee_count", "retail_subsector"],
        )
        extracted["employee_count"] = pd.to_numeric(
            extracted["employee_count"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(retail),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        ordered = extracted.assign(
            _company_sort=extracted["company_name"].astype(str).str.casefold()
        ).sort_values(
            ["employee_count", "_company_sort"],
            ascending=[False, True],
            na_position="last",
            kind="mergesort",
        )
        tracker.record("order_by", len(extracted), _intermediate_view(ordered))

        limited = ordered.head(3).reset_index(drop=True)
        tracker.record("limit", len(ordered), _intermediate_view(limited))

        answer = df_records(
            limited[["company_name", "employee_count", "retail_subsector"]]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
