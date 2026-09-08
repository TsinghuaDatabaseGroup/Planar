#!/usr/bin/env python3
"""Palimpzest pipeline for finance-002."""

from __future__ import annotations

import os
import re
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
    normalize_scalar_value,
    result_frame,
    save_output,
)

TASK_ID = "finance-002"
DATASET = "SEC"


def _normalize_mm_dd(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    numeric = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})", text)
    if numeric:
        month, day = map(int, numeric.groups())
        return f"{month:02d}-{day:02d}" if 1 <= month <= 12 and 1 <= day <= 31 else None
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%m-%d")


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

        extraction_plan = memory_dataset(
            f"{TASK_ID}-fiscal-year-end", documents
        ).sem_map(
            cols=[
                {
                    "name": "fiscal_year_end",
                    "type": str,
                    "desc": "The fiscal year-end date normalized to MM-DD format.",
                }
            ],
            desc="Extract and normalize the fiscal year-end date.",
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            ["fiscal_year_end"],
        )
        extracted["fiscal_year_end"] = extracted["fiscal_year_end"].map(
            _normalize_mm_dd
        )
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        non_december = extracted.loc[
            extracted["fiscal_year_end"].notna()
            & extracted["fiscal_year_end"].ne("12-31")
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), _intermediate_view(non_december))

        grouped = (
            non_december.groupby("fiscal_year_end")
            .size()
            .rename("count")
            .reset_index()
        )
        tracker.record("groupby", len(non_december), grouped)

        ordered = grouped.sort_values(
            ["count", "fiscal_year_end"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
