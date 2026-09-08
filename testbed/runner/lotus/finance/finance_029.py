#!/usr/bin/env python3
"""LOTUS pipeline for finance-029."""

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

TASK_ID = "finance-029"
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
        filings_2019 = load_table("SEC", "CSV/2019.csv")[["cik", "text"]].copy()
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2019.csv AS filings_2019)",
            None,
            len(filings_2019),
            output=filings_2019,
        )
        tracker.record(
            "SCAN_DOCS(selector=filings_2019.text)",
            len(filings_2019),
            len(filings_2019),
            output=filings_2019,
        )

        with tracker.step(
            "SEM_FILTER(challenger competitive position in 2019)",
            input_rows=len(filings_2019),
        ) as step:
            challenger_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings_2019,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                selected = documents.sem_filter(
                    "The 2019 10-K filing {document_text} describes the reporting "
                    "company as holding a challenger competitive position in its "
                    "industry or primary market in 2019."
                )
                challenger_parts.append(selected[["cik"]].copy())
            challengers_2019 = (
                pd.concat(challenger_parts, ignore_index=True)
                if challenger_parts
                else filings_2019.iloc[0:0][["cik"]].copy()
            )
            step.set_output(challengers_2019)

        tracker.record(
            "PROJECT(challengers_2019.cik)",
            len(challengers_2019),
            len(challengers_2019),
            output=challengers_2019,
        )

        filings_2023 = load_table("SEC", "CSV/2023.csv")[["cik", "text"]].copy()
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv AS filings_2023)",
            None,
            len(filings_2023),
            output=filings_2023,
        )
        tracker.record(
            "SCAN_DOCS(selector=filings_2023.text)",
            len(filings_2023),
            len(filings_2023),
            output=filings_2023,
        )

        with tracker.step(
            "SEM_FILTER(recognized industry or market leader in 2023)",
            input_rows=len(filings_2023),
        ) as step:
            leader_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings_2023,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                leader_parts.append(
                    documents.sem_filter(
                        "The 2023 10-K filing {document_text} describes the reporting "
                        "company as a recognized industry or market leader in 2023."
                    )
                )
            leaders_2023 = (
                pd.concat(leader_parts, ignore_index=True)
                if leader_parts
                else _empty_documents(filings_2023)
            )
            step.set_output(_intermediate_view(leaders_2023))

        deduped_leaders = leaders_2023.drop_duplicates("cik", keep="first").copy()
        tracker.record(
            "DEDUP(leaders_2023.cik, keep=first)",
            len(leaders_2023),
            len(deduped_leaders),
            output=_intermediate_view(deduped_leaders),
        )

        with tracker.step(
            "SEM_EXTRACT(2023 company legal name and industry sector)",
            input_rows=len(deduped_leaders),
        ) as step:
            extracted_leaders = sem_extract_in_batches(
                deduped_leaders,
                input_cols=["document_text"],
                output_cols={
                    "company_name": (
                        "the reporting company's legal name stated in the filing"
                    ),
                    "industry_sector": (
                        "the reporting company's industry sector stated or clearly "
                        "supported by the filing"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted_leaders["company_name"] = extracted_leaders[
                "company_name"
            ].map(clean_text)
            extracted_leaders["industry_sector"] = extracted_leaders[
                "industry_sector"
            ].map(clean_text)
            step.set_output(_intermediate_view(extracted_leaders))

        matched = challengers_2019.merge(
            extracted_leaders,
            on="cik",
            how="inner",
            sort=False,
        )
        tracker.record(
            "JOIN(inner, challengers_2019.cik = leaders_2023.cik)",
            {"left": len(challengers_2019), "right": len(extracted_leaders)},
            len(matched),
            output=_intermediate_view(matched),
        )

        answer_frame = matched[["company_name", "industry_sector"]].reset_index(
            drop=True
        )
        tracker.record(
            "PROJECT(company_name, industry_sector)",
            len(matched),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
