#!/usr/bin/env python3
"""LOTUS pipeline for finance-017."""

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

TASK_ID = "finance-017"
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
        filings = load_table("SEC", "CSV/2022.csv")[
            ["id", "cik", "text", "word_count"]
        ].copy()
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2022.csv)", None, len(filings), output=filings
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_FILTER(no substantive operations outside the United States)",
            input_rows=len(filings),
        ) as step:
            domestic_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                domestic_parts.append(
                    documents.sem_filter(
                        "The 10-K filing {document_text} does not describe substantive "
                        "business operations outside the United States."
                    )
                )
            domestic = (
                pd.concat(domestic_parts, ignore_index=True)
                if domestic_parts
                else _empty_documents(filings)
            )
            step.set_output(_intermediate_view(domestic))

        with tracker.step(
            "SEM_FILTER(environmental, social, or governance topics)",
            input_rows=len(domestic),
        ) as step:
            esg_filings = sem_filter_in_batches(
                domestic,
                "The 10-K filing {document_text} discusses environmental, social, or "
                "governance topics.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(esg_filings))

        with tracker.step(
            "SEM_EXTRACT(company legal name and industry sector)",
            input_rows=len(esg_filings),
        ) as step:
            extracted = sem_extract_in_batches(
                esg_filings,
                input_cols=["document_text"],
                output_cols={
                    "company_name": "the normalized legal name of the filing company",
                    "industry_sector": "the company's primary industry sector",
                },
                batch_size=BATCH_SIZE,
            )
            extracted["company_name"] = extracted["company_name"].map(clean_text)
            extracted["industry_sector"] = extracted["industry_sector"].map(clean_text)
            step.set_output(_intermediate_view(extracted))

        deduped = (
            extracted.assign(_company_key=extracted["company_name"].str.lower())
            .sort_values(
                ["_company_key", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("_company_key", keep="first")
            .drop(columns=["_company_key"])
        )
        tracker.record(
            "DEDUP(company_name, keep=max(word_count))",
            len(extracted),
            len(deduped),
            output=_intermediate_view(deduped),
        )

        ranked = deduped.assign(
            _company_sort=deduped["company_name"].str.lower()
        ).sort_values(
            ["word_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        ).head(5)
        answer_frame = ranked[
            ["company_name", "word_count", "industry_sector"]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(word_count DESC, company_name ASC) -> LIMIT(5)",
            len(deduped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
