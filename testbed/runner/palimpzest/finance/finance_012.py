#!/usr/bin/env python3
"""Palimpzest pipeline for finance-012."""

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

TASK_ID = "finance-012"
DATASET = "SEC"
AVAILABLE_YEARS = tuple(range(2019, 2025))


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
                for year in AVAILABLE_YEARS
            ],
            ignore_index=True,
        )
        filings["year"] = pd.to_numeric(
            filings["year"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        selected_years = filings.loc[
            filings["year"].isin([2022, 2023])
        ].reset_index(drop=True)
        tracker.record("filter", len(filings), selected_years)

        documents = load_selected_texts(
            DATASET,
            selected_years,
            path_column="text",
            output_column="document_text",
        )[["year", "document_text"]]
        tracker.record(
            "scan",
            len(selected_years),
            _intermediate_view(documents),
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-industry-ai", documents
        ).sem_map(
            cols=[
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": (
                        "The reporting company's concise industry sector stated or "
                        "clearly supported by the filing."
                    ),
                },
                {
                    "name": "ai_ml_mentioned",
                    "type": bool,
                    "desc": (
                        "True when the filing mentions artificial-intelligence or "
                        "machine-learning technologies; false otherwise."
                    ),
                },
            ],
            desc=(
                "Extract the industry sector and determine whether the filing "
                "mentions AI or machine-learning technologies."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            ["industry_sector", "ai_ml_mentioned"],
        )
        extracted["industry_sector"] = extracted["industry_sector"].map(
            lambda value: str(value).strip() if value is not None else ""
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

        grouped_rows = []
        for industry_sector, group in extracted.groupby(
            "industry_sector", sort=False
        ):
            is_2022 = group["year"].eq(2022)
            is_2023 = group["year"].eq(2023)
            grouped_rows.append(
                {
                    "industry_sector": industry_sector,
                    "ai_ml_filings_2022": int(
                        (is_2022 & group["ai_ml_mentioned"]).sum()
                    ),
                    "ai_ml_filings_2023": int(
                        (is_2023 & group["ai_ml_mentioned"]).sum()
                    ),
                    "total_filings_2023": int(is_2023.sum()),
                }
            )
        grouped = pd.DataFrame(
            grouped_rows,
            columns=[
                "industry_sector",
                "ai_ml_filings_2022",
                "ai_ml_filings_2023",
                "total_filings_2023",
            ],
        )
        tracker.record("groupby", len(extracted), grouped)

        qualifying = grouped.loc[
            grouped["total_filings_2023"].ge(20)
            & grouped["ai_ml_filings_2023"].gt(grouped["ai_ml_filings_2022"])
        ].copy()
        qualifying["_increase"] = (
            qualifying["ai_ml_filings_2023"]
            - qualifying["ai_ml_filings_2022"]
        )
        tracker.record(
            "filter",
            len(grouped),
            qualifying.drop(columns=["_increase"]),
        )

        ordered = qualifying.sort_values(
            ["_increase", "industry_sector"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record(
            "order_by",
            len(qualifying),
            ordered.drop(columns=["_increase"]),
        )

        limited = ordered.head(10).drop(columns=["_increase"]).reset_index(
            drop=True
        )
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
