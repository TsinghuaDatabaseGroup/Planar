#!/usr/bin/env python3
"""Palimpzest pipeline for finance-021."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "finance-021"
DATASET = "SEC"
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


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_2019", "document_2023"],
        errors="ignore",
    )


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings_2019 = load_table(
            DATASET,
            "CSV/2019.csv",
            ["cik", "text", "word_count"],
        )
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        filings_2019["word_count"] = pd.to_numeric(
            filings_2019["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2019)

        primary_2019 = (
            filings_2019.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .rename(columns={"text": "primary_text_2019"})
            [["cik", "primary_text_2019"]]
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(filings_2019), primary_2019)

        documents_2019 = load_selected_texts(
            DATASET,
            primary_2019,
            path_column="primary_text_2019",
            output_column="document_2019",
        )[["cik", "document_2019"]]
        tracker.record(
            "scan",
            len(primary_2019),
            _intermediate_view(documents_2019),
        )

        classification_2019_plan = memory_dataset(
            f"{TASK_ID}-revenue-2019", documents_2019
        ).sem_map(
            cols=[
                {
                    "name": "revenue_model_2019",
                    "type": str,
                    "desc": (
                        "Exactly one primary category: advertising, commission, "
                        "interest_income, investment_returns, licensing_royalties, "
                        "mixed, product_sales, rental_income, service_fees, "
                        "subscription, or undetermined."
                    ),
                }
            ],
            desc=(
                "Assign the 2019 filing to its primary revenue-model category "
                "according to how the reporting company primarily earns revenue."
            ),
            depends_on=["document_2019"],
        )
        started = time.time()
        classification_2019_result = classification_2019_plan.run(config)
        models_2019 = result_frame(
            classification_2019_result,
            documents_2019,
            ["revenue_model_2019"],
        )
        models_2019["revenue_model_2019"] = models_2019[
            "revenue_model_2019"
        ].map(lambda value: normalize_enum(value, REVENUE_MODELS))
        tracker.record_semantic(
            "sem_map",
            len(documents_2019),
            _intermediate_view(models_2019),
            classification_2019_result,
            time.time() - started,
        )

        filings_2023 = load_table(
            DATASET,
            "CSV/2023.csv",
            ["cik", "text", "word_count"],
        )
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        filings_2023["word_count"] = pd.to_numeric(
            filings_2023["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2023)

        primary_2023 = (
            filings_2023.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .rename(columns={"text": "primary_text_2023"})
            [["cik", "primary_text_2023"]]
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(filings_2023), primary_2023)

        documents_2023 = load_selected_texts(
            DATASET,
            primary_2023,
            path_column="primary_text_2023",
            output_column="document_2023",
        )[["cik", "document_2023"]]
        tracker.record(
            "scan",
            len(primary_2023),
            _intermediate_view(documents_2023),
        )

        classification_2023_plan = memory_dataset(
            f"{TASK_ID}-revenue-2023", documents_2023
        ).sem_map(
            cols=[
                {
                    "name": "revenue_model_2023",
                    "type": str,
                    "desc": (
                        "Exactly one primary category: advertising, commission, "
                        "interest_income, investment_returns, licensing_royalties, "
                        "mixed, product_sales, rental_income, service_fees, "
                        "subscription, or undetermined."
                    ),
                }
            ],
            desc=(
                "Assign the 2023 filing to its primary revenue-model category "
                "according to how the reporting company primarily earns revenue."
            ),
            depends_on=["document_2023"],
        )
        started = time.time()
        classification_2023_result = classification_2023_plan.run(config)
        models_2023 = result_frame(
            classification_2023_result,
            documents_2023,
            ["revenue_model_2023"],
        )
        models_2023["revenue_model_2023"] = models_2023[
            "revenue_model_2023"
        ].map(lambda value: normalize_enum(value, REVENUE_MODELS))
        tracker.record_semantic(
            "sem_map",
            len(documents_2023),
            _intermediate_view(models_2023),
            classification_2023_result,
            time.time() - started,
        )

        paired = models_2019.loc[models_2019["cik"].notna()].merge(
            models_2023.loc[models_2023["cik"].notna()],
            on="cik",
            how="inner",
            sort=False,
        )
        tracker.record(
            "join",
            {"left": len(models_2019), "right": len(models_2023)},
            _intermediate_view(paired),
        )

        changed = paired.loc[
            paired["revenue_model_2019"].notna()
            & paired["revenue_model_2023"].notna()
            & paired["revenue_model_2019"].ne("undetermined")
            & paired["revenue_model_2023"].ne("undetermined")
            & paired["revenue_model_2019"].ne(paired["revenue_model_2023"])
        ].reset_index(drop=True)
        tracker.record("filter", len(paired), _intermediate_view(changed))

        transitions = pd.DataFrame(
            {
                "transition": (
                    changed["revenue_model_2019"]
                    + " -> "
                    + changed["revenue_model_2023"]
                )
            }
        )
        tracker.record("project", len(changed), transitions)

        grouped = (
            transitions.groupby("transition", sort=False)
            .size()
            .rename("count")
            .reset_index()
        )
        tracker.record("groupby", len(transitions), grouped)

        ordered = grouped.sort_values(
            ["count", "transition"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(grouped), ordered)

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
