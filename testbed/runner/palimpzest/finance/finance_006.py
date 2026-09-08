#!/usr/bin/env python3
"""Palimpzest pipeline for finance-006."""

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
    result_frame,
    save_output,
)

TASK_ID = "finance-006"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def _capped_count(value, maximum: int) -> int:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return 0
    return min(max(int(parsed), 0), maximum)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2022.csv",
            ["cik", "text", "word_count"],
        )
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["cik", "word_count", "document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-standardized-review", documents
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The reporting company's legal name.",
                },
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": "The reporting company's industry sector.",
                },
                {
                    "name": "subsidiary_count",
                    "type": int,
                    "desc": (
                        "The number of distinct specifically named legal "
                        "subsidiaries disclosed in the filing, excluding the "
                        "registrant itself and capped at ten; an integer from 0 "
                        "through 10."
                    ),
                },
                {
                    "name": "competitive_advantages_count",
                    "type": int,
                    "desc": (
                        "The number of distinct competitive-advantage themes "
                        "described for the reporting company, capped at eight; an "
                        "integer from 0 through 8."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name and industry sector, and record "
                "the capped counts of named subsidiaries and distinct competitive-"
                "advantage themes."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            [
                "company_name",
                "industry_sector",
                "subsidiary_count",
                "competitive_advantages_count",
            ],
        )
        extracted["subsidiary_count"] = extracted["subsidiary_count"].map(
            lambda value: _capped_count(value, 10)
        )
        extracted["competitive_advantages_count"] = extracted[
            "competitive_advantages_count"
        ].map(lambda value: _capped_count(value, 8))
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["subsidiary_count"].ge(8)
            & extracted["competitive_advantages_count"].ge(6)
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(extracted),
            _intermediate_view(qualifying),
        )

        qualifying["combined_count"] = (
            qualifying["competitive_advantages_count"]
            + qualifying["subsidiary_count"]
        )
        selected = (
            qualifying.sort_values(
                ["cik", "combined_count", "word_count"],
                ascending=[True, False, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(qualifying), _intermediate_view(selected))

        ordered = selected.sort_values(
            ["combined_count", "company_name"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(selected), _intermediate_view(ordered))

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), _intermediate_view(limited))

        answer_frame = limited[
            [
                "company_name",
                "competitive_advantages_count",
                "subsidiary_count",
                "industry_sector",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
