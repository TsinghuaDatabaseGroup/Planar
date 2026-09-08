#!/usr/bin/env python3
"""Palimpzest pipeline for finance-009."""

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

TASK_ID = "finance-009"
DATASET = "SEC"
COMPETITIVE_POSITIONS = ("market_leader", "other")


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def _capped_advantage_count(value) -> int:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return 0
    return min(max(int(parsed), 0), 8)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(DATASET, "CSV/2023.csv", ["text"])
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-advantages", documents
        ).sem_map(
            cols=[
                {
                    "name": "competitive_advantage_count",
                    "type": int,
                    "desc": (
                        "The number of distinct competitive-advantage themes "
                        "described for the reporting company, capped at eight; an "
                        "integer from 0 through 8."
                    ),
                },
                {
                    "name": "competitive_position",
                    "type": str,
                    "desc": (
                        "market_leader when the filing describes the reporting "
                        "company as a market leader, or other otherwise."
                    ),
                },
            ],
            desc=(
                "Record up to eight distinct competitive-advantage themes and "
                "classify the company's competitive position."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            ["competitive_advantage_count", "competitive_position"],
        )
        extracted["competitive_advantage_count"] = extracted[
            "competitive_advantage_count"
        ].map(_capped_advantage_count)
        extracted["competitive_position"] = extracted[
            "competitive_position"
        ].map(lambda value: normalize_enum(value, COMPETITIVE_POSITIONS))
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        projected = pd.DataFrame(
            {
                "advantage_group": extracted[
                    "competitive_advantage_count"
                ].map(
                    lambda value: (
                        "high_advantage_group"
                        if value == 8
                        else "low_advantage_group"
                    )
                ),
                "is_market_leader": extracted["competitive_position"].eq(
                    "market_leader"
                ),
            }
        )
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("advantage_group", sort=False)
            .agg(
                filing_count=("advantage_group", "size"),
                market_leader_rate_percent=("is_market_leader", "mean"),
            )
            .reset_index()
        )
        grouped["market_leader_rate_percent"] = (
            100.0 * grouped["market_leader_rate_percent"]
        ).round(1)
        tracker.record("groupby", len(projected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
