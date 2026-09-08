#!/usr/bin/env python3
"""LOTUS pipeline for finance-025."""

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
    parse_number,
    save_output,
    sem_extract_in_batches,
    sem_filter_in_batches,
    setup,
)

TASK_ID = "finance-025"
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
        filings = load_table("SEC", "CSV/2019.csv")[
            ["id", "cik", "text", "word_count"]
        ].copy()
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2019.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 45000].copy()
        tracker.record(
            "FILTER(word_count > 45000)",
            len(filings),
            len(candidates),
            output=candidates,
        )
        tracker.record(
            "SCAN_DOCS(selector=filtered.text)",
            len(candidates),
            len(candidates),
            output=candidates,
        )

        with tracker.step(
            "SEM_FILTER(dependence on a major customer)",
            input_rows=len(candidates),
        ) as step:
            customer_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                customer_parts.append(
                    documents.sem_filter(
                        "The 10-K filing {document_text} discloses dependence on a "
                        "major customer or material customer concentration."
                    )
                )
            customer_filings = (
                pd.concat(customer_parts, ignore_index=True)
                if customer_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(customer_filings))

        with tracker.step(
            "SEM_FILTER(no foreign-currency or exchange-rate risk disclosure)",
            input_rows=len(customer_filings),
        ) as step:
            no_currency_risk = sem_filter_in_batches(
                customer_filings,
                "The 10-K filing {document_text} does not disclose foreign-currency "
                "risk or exchange-rate risk.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(no_currency_risk))

        with tracker.step(
            "SEM_FILTER(operations outside the United States)",
            input_rows=len(no_currency_risk),
        ) as step:
            international = sem_filter_in_batches(
                no_currency_risk,
                "The 10-K filing {document_text} describes company operations outside "
                "the United States.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(international))

        with tracker.step(
            "SEM_EXTRACT(company name, industry sector, employee count)",
            input_rows=len(international),
        ) as step:
            extracted = sem_extract_in_batches(
                international,
                input_cols=["document_text"],
                output_cols={
                    "company_name": "the company's legal name as reported in the filing",
                    "industry_sector": "the company's primary industry sector",
                    "employee_count": (
                        "the total reported employee count as an integer, or null when "
                        "the filing does not report a determinable total"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted["company_name"] = extracted["company_name"].map(clean_text)
            extracted["industry_sector"] = extracted["industry_sector"].map(clean_text)
            extracted["employee_count"] = pd.to_numeric(
                extracted["employee_count"].map(parse_number), errors="coerce"
            ).astype("Int64")
            step.set_output(_intermediate_view(extracted))

        valid = extracted[extracted["employee_count"].notna()].copy()
        tracker.record(
            "FILTER(employee_count IS NOT NULL)",
            len(extracted),
            len(valid),
            output=_intermediate_view(valid),
        )

        deduped = (
            valid.assign(_company_key=valid["company_name"].str.lower())
            .sort_values(
                ["_company_key", "employee_count", "word_count", "cik"],
                ascending=[True, False, False, False],
                kind="mergesort",
            )
            .drop_duplicates("_company_key", keep="first")
            .drop(columns=["_company_key"])
        )
        tracker.record(
            "DEDUP(company_name, max_by(employee_count, word_count, cik))",
            len(valid),
            len(deduped),
            output=_intermediate_view(deduped),
        )

        ranked = deduped.assign(
            _company_sort=deduped["company_name"].str.lower()
        ).sort_values(
            ["employee_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)
        answer_frame = ranked[
            ["company_name", "industry_sector", "employee_count"]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(employee_count DESC, LOWER(company_name) ASC) -> LIMIT(10) -> PROJECT",
            len(deduped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
