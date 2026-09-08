#!/usr/bin/env python3
"""LOTUS pipeline for finance-004."""

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
    save_output,
    setup,
)

TASK_ID = "finance-004"
BATCH_SIZE = 100
REVENUE_MODELS = (
    "mixed",
    "product_sales",
    "interest_income",
    "service_fees",
    "licensing_royalties",
    "rental_income",
    "investment_returns",
    "subscription",
    "commission",
    "undetermined",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2023.csv")[["text", "word_count"]].copy()
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 100000].copy()
        tracker.record(
            "FILTER(word_count > 100000)",
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
            "SEM_CLASSIFY(primary revenue-model category)",
            input_rows=len(candidates),
        ) as step:
            classified_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_map(
                    "Assign the 2023 10-K filing {document_text} to exactly one "
                    "primary revenue-model category according to how the reporting "
                    "company primarily earns revenue. Output exactly one label: "
                    "mixed, product_sales, interest_income, service_fees, "
                    "licensing_royalties, rental_income, investment_returns, "
                    "subscription, commission, or undetermined when no primary "
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
                else candidates.iloc[0:0].assign(
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

        answer_frame = (
            determined.groupby("revenue_model", sort=False)
            .size()
            .rename("filing_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(revenue_model, COUNT(*) AS filing_count)",
            len(determined),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
