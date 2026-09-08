#!/usr/bin/env python3
"""LOTUS pipeline for finance-021."""

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

TASK_ID = "finance-021"
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
        filings_2019 = load_table("SEC", "CSV/2019.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        filings_2019["word_count"] = pd.to_numeric(
            filings_2019["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2019.csv AS filings_2019)",
            None,
            len(filings_2019),
            output=filings_2019,
        )

        primary_2019 = (
            filings_2019.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .rename(columns={"text": "primary_text_2019"})[
                ["cik", "primary_text_2019"]
            ]
            .copy()
        )
        tracker.record(
            "GROUP_BY(cik, MAX_BY(text, word_count) AS primary_text_2019)",
            len(filings_2019),
            len(primary_2019),
            output=primary_2019,
        )
        tracker.record(
            "SCAN_DOCS(selector=primary_text_2019)",
            len(primary_2019),
            len(primary_2019),
            output=primary_2019,
        )

        with tracker.step(
            "SEM_CLASSIFY(2019 primary revenue-model category)",
            input_rows=len(primary_2019),
        ) as step:
            model_2019_parts = []
            for documents in iter_selected_texts(
                "SEC",
                primary_2019,
                path_column="primary_text_2019",
                output_column="document_2019",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_map(
                    "Assign the 2019 10-K filing {document_2019} to exactly one "
                    "primary revenue-model category according to how the reporting "
                    "company primarily earns revenue. Output exactly one label: "
                    "advertising, commission, interest_income, investment_returns, "
                    "licensing_royalties, mixed, product_sales, rental_income, "
                    "service_fees, subscription, or undetermined.",
                    suffix="revenue_model_2019",
                )
                batch["revenue_model_2019"] = batch[
                    "revenue_model_2019"
                ].map(lambda value: normalize_enum(value, REVENUE_MODELS))
                model_2019_parts.append(
                    batch.drop(columns=["document_2019"], errors="ignore")
                )
            models_2019 = (
                pd.concat(model_2019_parts, ignore_index=True)
                if model_2019_parts
                else primary_2019.iloc[0:0].assign(
                    revenue_model_2019=pd.Series(dtype="object")
                )
            )
            step.set_output(models_2019)

        filings_2023 = load_table("SEC", "CSV/2023.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        filings_2023["word_count"] = pd.to_numeric(
            filings_2023["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv AS filings_2023)",
            None,
            len(filings_2023),
            output=filings_2023,
        )

        primary_2023 = (
            filings_2023.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .rename(columns={"text": "primary_text_2023"})[
                ["cik", "primary_text_2023"]
            ]
            .copy()
        )
        tracker.record(
            "GROUP_BY(cik, MAX_BY(text, word_count) AS primary_text_2023)",
            len(filings_2023),
            len(primary_2023),
            output=primary_2023,
        )
        tracker.record(
            "SCAN_DOCS(selector=primary_text_2023)",
            len(primary_2023),
            len(primary_2023),
            output=primary_2023,
        )

        with tracker.step(
            "SEM_CLASSIFY(2023 primary revenue-model category)",
            input_rows=len(primary_2023),
        ) as step:
            model_2023_parts = []
            for documents in iter_selected_texts(
                "SEC",
                primary_2023,
                path_column="primary_text_2023",
                output_column="document_2023",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_map(
                    "Assign the 2023 10-K filing {document_2023} to exactly one "
                    "primary revenue-model category according to how the reporting "
                    "company primarily earns revenue. Output exactly one label: "
                    "advertising, commission, interest_income, investment_returns, "
                    "licensing_royalties, mixed, product_sales, rental_income, "
                    "service_fees, subscription, or undetermined.",
                    suffix="revenue_model_2023",
                )
                batch["revenue_model_2023"] = batch[
                    "revenue_model_2023"
                ].map(lambda value: normalize_enum(value, REVENUE_MODELS))
                model_2023_parts.append(
                    batch.drop(columns=["document_2023"], errors="ignore")
                )
            models_2023 = (
                pd.concat(model_2023_parts, ignore_index=True)
                if model_2023_parts
                else primary_2023.iloc[0:0].assign(
                    revenue_model_2023=pd.Series(dtype="object")
                )
            )
            step.set_output(models_2023)

        paired = models_2019.merge(
            models_2023,
            on="cik",
            how="inner",
            sort=False,
        )
        tracker.record(
            "JOIN(inner, models_2019.cik = models_2023.cik)",
            {"left": len(models_2019), "right": len(models_2023)},
            len(paired),
            output=paired,
        )

        changed = paired[
            paired["revenue_model_2019"].notna()
            & paired["revenue_model_2023"].notna()
            & (paired["revenue_model_2019"] != "undetermined")
            & (paired["revenue_model_2023"] != "undetermined")
            & (paired["revenue_model_2019"] != paired["revenue_model_2023"])
        ].copy()
        tracker.record(
            "FILTER(both models determined AND revenue_model_2019 != revenue_model_2023)",
            len(paired),
            len(changed),
            output=changed,
        )

        transitions = pd.DataFrame(
            {
                "transition": (
                    changed["revenue_model_2019"]
                    + " -> "
                    + changed["revenue_model_2023"]
                )
            }
        )
        tracker.record(
            "PROJECT(revenue_model_2019 || ' -> ' || revenue_model_2023 AS transition)",
            len(changed),
            len(transitions),
            output=transitions,
        )

        grouped = (
            transitions.groupby("transition", sort=False)
            .size()
            .rename("count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(transition, COUNT(*) AS count)",
            len(transitions),
            len(grouped),
            output=grouped,
        )

        answer_frame = grouped.sort_values(
            ["count", "transition"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(count DESC, transition ASC) -> LIMIT(10)",
            len(grouped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} transition rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
