#!/usr/bin/env python3
"""LOTUS pipeline for finance-002."""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    save_output,
    sem_extract_in_batches,
    setup,
)

TASK_ID = "finance-002"
BATCH_SIZE = 100


def _normalize_mm_dd(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    numeric = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})", text)
    if numeric:
        month, day = map(int, numeric.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{month:02d}-{day:02d}"
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%m-%d")


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2024.csv")[["id", "cik", "text"]].copy()
        tracker.record(
            "SCAN_TABLE(CSV/2024.csv)", None, len(filings), output=filings
        )

        documents = load_selected_texts(
            "SEC",
            filings,
            path_column="text",
            output_column="document_text",
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(documents),
            output=filings,
        )

        with tracker.step(
            "SEM_EXTRACT(fiscal year-end date in MM-DD format)",
            input_rows=len(documents),
        ) as step:
            extracted = sem_extract_in_batches(
                documents,
                input_cols=["document_text"],
                output_cols={
                    "fiscal_year_end": (
                        "the fiscal year-end date, normalized to MM-DD format"
                    )
                },
                batch_size=BATCH_SIZE,
            )
            extracted["fiscal_year_end"] = extracted["fiscal_year_end"].map(
                _normalize_mm_dd
            )
            step.set_output(extracted.drop(columns=["document_text"], errors="ignore"))

        non_december = extracted[
            extracted["fiscal_year_end"].notna()
            & (extracted["fiscal_year_end"] != "12-31")
        ].copy()
        tracker.record(
            "FILTER(fiscal_year_end != '12-31')",
            len(extracted),
            len(non_december),
            output=non_december.drop(columns=["document_text"], errors="ignore"),
        )

        grouped = (
            non_december.groupby("fiscal_year_end")
            .size()
            .rename("count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(fiscal_year_end, COUNT(*))",
            len(non_december),
            len(grouped),
            output=grouped,
        )

        answer_frame = grouped.sort_values(
            ["count", "fiscal_year_end"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(count DESC, fiscal_year_end ASC)",
            len(grouped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
