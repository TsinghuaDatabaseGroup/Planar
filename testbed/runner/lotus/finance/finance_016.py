#!/usr/bin/env python3
"""LOTUS pipeline for finance-016."""

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
    setup,
)

TASK_ID = "finance-016"
BATCH_SIZE = 100


def _mentions_ai_or_ml(texts: pd.Series) -> pd.Series:
    return texts.str.contains(
        "artificial intelligence", case=False, regex=False, na=False
    ) | texts.str.contains("machine learning", case=False, regex=False, na=False)


def _intermediate_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=["document_2019", "document_2023"], errors="ignore"
    )


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings_2019 = load_table("SEC", "CSV/2019.csv")[
            ["id", "cik", "text"]
        ].rename(columns={"id": "id_2019", "text": "text_2019"})
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2019.csv AS filings_2019)",
            None,
            len(filings_2019),
            output=filings_2019,
        )

        filings_2023 = load_table("SEC", "CSV/2023.csv")[
            ["id", "cik", "text"]
        ].rename(columns={"id": "id_2023", "text": "text_2023"})
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv AS filings_2023)",
            None,
            len(filings_2023),
            output=filings_2023,
        )

        joined = filings_2019.merge(filings_2023, on="cik", how="inner", sort=False)
        tracker.record(
            "JOIN(inner, filings_2019.cik = filings_2023.cik)",
            {"left": len(filings_2019), "right": len(filings_2023)},
            len(joined),
            output=joined,
        )
        tracker.record(
            "SCAN_DOCS(selector=text_2023)",
            len(joined),
            len(joined),
            output=joined,
        )

        mentions_2023_parts = []
        for documents in iter_selected_texts(
            "SEC",
            joined,
            path_column="text_2023",
            output_column="document_2023",
            batch_size=BATCH_SIZE,
        ):
            mentions_2023_parts.append(
                documents[_mentions_ai_or_ml(documents["document_2023"])].copy()
            )
        mentions_2023 = (
            pd.concat(mentions_2023_parts, ignore_index=True)
            if mentions_2023_parts
            else joined.iloc[0:0].assign(
                document_2023=pd.Series(dtype="object")
            )
        )
        tracker.record(
            "FILTER(document_2023 CONTAINS_CI 'artificial intelligence' OR 'machine learning')",
            len(joined),
            len(mentions_2023),
            output=_intermediate_view(mentions_2023),
        )

        deduped = mentions_2023.drop_duplicates("cik", keep="first").copy()
        tracker.record(
            "DEDUP(cik, keep=first)",
            len(mentions_2023),
            len(deduped),
            output=_intermediate_view(deduped),
        )
        tracker.record(
            "SCAN_DOCS(selector=text_2019)",
            len(deduped),
            len(deduped),
            output=_intermediate_view(deduped),
        )

        no_mentions_2019_parts = []
        for documents in iter_selected_texts(
            "SEC",
            deduped,
            path_column="text_2019",
            output_column="document_2019",
            batch_size=BATCH_SIZE,
        ):
            no_mentions_2019_parts.append(
                documents[~_mentions_ai_or_ml(documents["document_2019"])].copy()
            )
        no_mentions_2019 = (
            pd.concat(no_mentions_2019_parts, ignore_index=True)
            if no_mentions_2019_parts
            else deduped.iloc[0:0].assign(
                document_2019=pd.Series(dtype="object")
            )
        )
        tracker.record(
            "FILTER(NOT document_2019 CONTAINS_CI 'artificial intelligence' AND NOT 'machine learning')",
            len(deduped),
            len(no_mentions_2019),
            output=_intermediate_view(no_mentions_2019),
        )

        with tracker.step(
            "SEM_EXTRACT(2023 industry sector)",
            input_rows=len(no_mentions_2019),
        ) as step:
            extracted = sem_extract_in_batches(
                no_mentions_2019,
                input_cols=["document_2023"],
                output_cols={
                    "industry_sector": "the company's primary industry sector in 2023"
                },
                batch_size=BATCH_SIZE,
            )
            extracted["industry_sector"] = extracted["industry_sector"].map(
                clean_text
            )
            step.set_output(_intermediate_view(extracted))

        grouped = (
            extracted.groupby("industry_sector")
            .size()
            .rename("company_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(industry_sector, COUNT(*))",
            len(extracted),
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
