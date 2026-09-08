#!/usr/bin/env python3
"""Palimpzest pipeline for finance-007."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "finance-007"
DATASET = "SEC"
YEARS = tuple(range(2019, 2025))


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = pd.concat(
            [
                load_table(
                    DATASET,
                    f"CSV/{year}.csv",
                    ["year", "text"],
                )
                for year in YEARS
            ],
            ignore_index=True,
        )
        filings["year"] = pd.to_numeric(
            filings["year"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["year", "document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-ai-ml-mentions", documents
        ).sem_map(
            cols=[
                {
                    "name": "ai_ml_mentioned",
                    "type": bool,
                    "desc": (
                        "True only when the filing specifically mentions artificial "
                        "intelligence, machine learning, deep learning, neural "
                        "networks, AI-powered systems, or a similar AI/ML technology "
                        "in the context of the reporting company's business, "
                        "products, or strategy; false otherwise."
                    ),
                }
            ],
            desc=(
                "Determine whether the filing contains a qualifying AI/ML "
                "technology mention."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            ["ai_ml_mentioned"],
        )
        extracted["ai_ml_mentioned"] = extracted["ai_ml_mentioned"].map(
            parse_bool
        )
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("year", sort=False)
            .agg(
                filing_count=("year", "size"),
                ai_ml_mentions=("ai_ml_mentioned", "sum"),
            )
            .reset_index()
        )
        grouped["ai_ml_mention_rate_percent"] = (
            100.0 * grouped["ai_ml_mentions"] / grouped["filing_count"]
        ).round(1)
        grouped = grouped[["year", "ai_ml_mention_rate_percent"]]
        tracker.record("groupby", len(extracted), grouped)

        ordered = grouped.sort_values(
            "year", ascending=True, kind="mergesort"
        ).reset_index(drop=True)
        tracker.record("order_by", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
