#!/usr/bin/env python3
"""LOTUS pipeline for finance-034."""

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

TASK_ID = "finance-034"
BATCH_SIZE = 100


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _intermediate_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["document_text"], errors="ignore")


def _clean_nullable_text(value) -> str | None:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    text = clean_text(value, default="")
    return text or None


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2023.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 60000].copy()
        tracker.record(
            "FILTER(word_count > 60000)",
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
            "SEM_FILTER(exclusively domestic operating footprint)",
            input_rows=len(candidates),
        ) as step:
            domestic_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                domestic_parts.append(
                    documents.sem_filter(
                        "The 2023 10-K filing {document_text} describes the reporting "
                        "company's operating footprint as exclusively domestic, with "
                        "no operations outside the United States."
                    )
                )
            domestic = (
                pd.concat(domestic_parts, ignore_index=True)
                if domestic_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(domestic))

        with tracker.step(
            "SEM_FILTER(foreign-currency or exchange-rate risk)",
            input_rows=len(domestic),
        ) as step:
            currency_risk = sem_filter_in_batches(
                domestic,
                "The 2023 10-K filing {document_text} discloses foreign-currency or "
                "exchange-rate risk for the reporting company.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(currency_risk))

        with tracker.step(
            "SEM_EXTRACT(company, industry, and total risk-factor count)",
            input_rows=len(currency_risk),
        ) as step:
            extracted = sem_extract_in_batches(
                currency_risk,
                input_cols=["document_text"],
                output_cols={
                    "company_name": (
                        "the reporting company's legal name stated in the filing, "
                        "or null when it cannot be determined"
                    ),
                    "industry_sector": (
                        "the reporting company's industry sector stated or clearly "
                        "supported by the filing, or null when it cannot be determined"
                    ),
                    "risk_factor_count": (
                        "the total number of distinct risk factors in the filing's "
                        "Risk Factors section as an integer, or null when the total "
                        "cannot be determined"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted["company_name"] = extracted["company_name"].map(
                _clean_nullable_text
            )
            extracted["industry_sector"] = extracted["industry_sector"].map(
                _clean_nullable_text
            )
            extracted["risk_factor_count"] = pd.to_numeric(
                extracted["risk_factor_count"].map(parse_number), errors="coerce"
            )
            step.set_output(_intermediate_view(extracted))

        complete = extracted[
            extracted["company_name"].notna()
            & extracted["industry_sector"].notna()
            & extracted["risk_factor_count"].notna()
        ].copy()
        tracker.record(
            "FILTER(company_name IS NOT NULL AND industry_sector IS NOT NULL AND risk_factor_count IS NOT NULL)",
            len(extracted),
            len(complete),
            output=_intermediate_view(complete),
        )

        deduped = (
            complete.sort_values(
                ["company_name", "risk_factor_count", "word_count", "cik"],
                ascending=[True, False, False, False],
                kind="mergesort",
            )
            .drop_duplicates("company_name", keep="first")
            .copy()
        )
        tracker.record(
            "DEDUP(company_name, keep=max_by(risk_factor_count, word_count, cik))",
            len(complete),
            len(deduped),
            output=_intermediate_view(deduped),
        )

        ranked = deduped.assign(
            _company_sort=deduped["company_name"].str.lower()
        ).sort_values(
            ["risk_factor_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)
        answer_frame = ranked[
            ["company_name", "industry_sector", "risk_factor_count"]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(risk_factor_count DESC, LOWER(company_name) ASC) -> LIMIT(10) -> PROJECT",
            len(deduped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
