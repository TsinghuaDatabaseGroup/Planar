#!/usr/bin/env python3
"""Palimpzest pipeline for finance-008."""

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

TASK_ID = "finance-008"
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
            filings["year"].between(2019, 2023)
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
            f"{TASK_ID}-going-concern", documents
        ).sem_map(
            cols=[
                {
                    "name": "has_going_concern",
                    "type": bool,
                    "desc": (
                        "True only when the filing expresses substantial doubt "
                        "about the reporting company's ability to continue as a "
                        "going concern; false otherwise."
                    ),
                }
            ],
            desc=(
                "Determine whether the filing expresses substantial doubt about "
                "the company's ability to continue as a going concern."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            ["has_going_concern"],
        )
        extracted["has_going_concern"] = extracted["has_going_concern"].map(
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
                going_concern_filings=("has_going_concern", "sum"),
            )
            .reset_index()
        )
        grouped["going_concern_rate_percent"] = (
            100.0
            * grouped["going_concern_filings"]
            / grouped["filing_count"]
        ).round(1)
        grouped = grouped[["year", "going_concern_rate_percent"]]
        tracker.record("groupby", len(extracted), grouped)

        ordered = grouped.sort_values(
            "year", ascending=True, kind="mergesort"
        ).reset_index(drop=True)
        tracker.record("order_by", len(grouped), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
