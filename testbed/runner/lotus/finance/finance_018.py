#!/usr/bin/env python3
"""LOTUS pipeline for finance-018."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    iter_selected_texts,
    load_table,
    normalize_enum,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "finance-018"
BATCH_SIZE = 100
REVENUE_MODELS = (
    "advertising",
    "commission",
    "interest_income",
    "investment_returns",
    "licensing_royalties",
    "mixed",
    "product_sales",
    "rental_income",
    "service_fees",
    "subscription",
    "undetermined",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2023.csv")[["text"]].copy()
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv)", None, len(filings), output=filings
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_CLASSIFY(primary revenue-model category)",
            input_rows=len(filings),
        ) as step:
            classified_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_map(
                    "Assign the 2023 10-K filing {document_text} to exactly one "
                    "primary revenue-model category according to how the reporting "
                    "company primarily earns revenue. Output exactly one label: "
                    "advertising, commission, interest_income, investment_returns, "
                    "licensing_royalties, mixed, product_sales, rental_income, "
                    "service_fees, subscription, or undetermined when no primary "
                    "model can be determined.",
                    suffix="revenue_model",
                )
                batch["revenue_model"] = batch["revenue_model"].map(
                    lambda value: normalize_enum(value, REVENUE_MODELS)
                )
                classified_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            classified = (
                pd.concat(classified_parts, ignore_index=True)
                if classified_parts
                else filings.iloc[0:0].assign(
                    revenue_model=pd.Series(dtype="object")
                )
            )
            step.set_output(classified)

        determined = classified[
            classified["revenue_model"].notna()
            & (classified["revenue_model"] != "undetermined")
        ].copy()
        tracker.record(
            "FILTER(revenue_model IS NOT NULL)",
            len(classified),
            len(determined),
            output=determined,
        )

        with tracker.step(
            "SEM_EXTRACT(reported employee count)",
            input_rows=len(determined),
        ) as step:
            extracted_parts = []
            for documents in iter_selected_texts(
                "SEC",
                determined,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["document_text"],
                    output_cols={
                        "employee_count": (
                            "the explicitly reported company-wide total employee "
                            "count as an integer, or null when no explicit total is "
                            "disclosed"
                        )
                    },
                )
                batch["employee_count"] = pd.to_numeric(
                    batch["employee_count"].map(parse_number), errors="coerce"
                )
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else determined.iloc[0:0].assign(
                    employee_count=pd.Series(dtype="float64")
                )
            )
            step.set_output(extracted)

        answer_frame = (
            extracted.groupby("revenue_model", sort=False)
            .agg(
                filing_count=("revenue_model", "size"),
                average_employee_count=("employee_count", "mean"),
            )
            .reset_index()
        )
        answer_frame["average_employee_count"] = pd.to_numeric(
            answer_frame["average_employee_count"], errors="coerce"
        ).round(0)
        tracker.record(
            "GROUP_BY(revenue_model, COUNT(*), ROUND(AVG(employee_count IGNORE NULLS), 0))",
            len(extracted),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
