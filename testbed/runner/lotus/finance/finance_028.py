#!/usr/bin/env python3
"""LOTUS pipeline for finance-028."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    iter_selected_texts,
    load_table,
    save_output,
    sem_extract_in_batches,
    sem_filter_in_batches,
    setup,
)

TASK_ID = "finance-028"
BATCH_SIZE = 100


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
        filings = load_table("SEC", "CSV/2021.csv")[["id", "cik", "text"]].copy()
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        tracker.record(
            "SCAN_TABLE(CSV/2021.csv)", None, len(filings), output=filings
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_FILTER(substantial doubt about continuing as a going concern)",
            input_rows=len(filings),
        ) as step:
            concern_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                concern_parts.append(
                    documents.sem_filter(
                        "The 10-K filing {document_text} reports substantial doubt "
                        "about the company's ability to continue as a going concern."
                    )
                )
            concern_filings = (
                pd.concat(concern_parts, ignore_index=True)
                if concern_parts
                else _empty_documents(filings)
            )
            step.set_output(_intermediate_view(concern_filings))

        with tracker.step(
            "SEM_FILTER(pending litigation)",
            input_rows=len(concern_filings),
        ) as step:
            litigation = sem_filter_in_batches(
                concern_filings,
                "The 10-K filing {document_text} reports pending litigation.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(litigation))

        with tracker.step(
            "SEM_FILTER(single customer accounts for at least 10 percent of revenue)",
            input_rows=len(litigation),
        ) as step:
            concentrated = sem_filter_in_batches(
                litigation,
                "The 10-K filing {document_text} reports that a single customer "
                "accounts for at least 10 percent of revenue.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(concentrated))

        with tracker.step(
            "SEM_EXTRACT(industry sector)",
            input_rows=len(concentrated),
        ) as step:
            extracted = sem_extract_in_batches(
                concentrated,
                input_cols=["document_text"],
                output_cols={
                    "industry_sector": "the company's primary industry sector"
                },
                batch_size=BATCH_SIZE,
            )
            extracted["industry_sector"] = extracted["industry_sector"].map(
                clean_text
            )
            step.set_output(_intermediate_view(extracted))

        deduped = extracted.drop_duplicates("cik", keep="first").copy()
        tracker.record(
            "DEDUP(cik, keep=first)",
            len(extracted),
            len(deduped),
            output=_intermediate_view(deduped),
        )

        grouped = (
            deduped.groupby("industry_sector")
            .size()
            .rename("company_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(industry_sector, COUNT(*))",
            len(deduped),
            len(grouped),
            output=grouped,
        )

        answer_frame = grouped.sort_values(
            ["company_count", "industry_sector"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(company_count DESC, industry_sector ASC) -> LIMIT(10)",
            len(grouped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
