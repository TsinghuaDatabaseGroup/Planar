#!/usr/bin/env python3
"""Palimpzest pipeline for finance-003."""

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

TASK_ID = "finance-003"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2023.csv",
            ["id", "cik", "text"],
        )
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["id", "cik", "document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-risk-factors", documents
        ).sem_map(
            cols=[
                {
                    "name": "industry",
                    "type": str,
                    "desc": "The company's primary industry sector.",
                },
                {
                    "name": "risk_factor_count",
                    "type": int,
                    "desc": (
                        "The number of distinct risk factors listed in Item 1A as "
                        "an integer, or null when the count cannot be determined "
                        "reliably."
                    ),
                },
            ],
            desc=(
                "Extract the primary industry sector and count the distinct risk "
                "factors listed in Item 1A."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            ["industry", "risk_factor_count"],
        )
        extracted["risk_factor_count"] = pd.to_numeric(
            extracted["risk_factor_count"], errors="coerce"
        )
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("industry")
            .agg(
                avg_risk_factor_count=("risk_factor_count", "mean"),
                filing_count=("risk_factor_count", "count"),
            )
            .reset_index()
        )
        grouped["avg_risk_factor_count"] = grouped[
            "avg_risk_factor_count"
        ].round(1)
        tracker.record("groupby", len(extracted), grouped)

        qualifying = grouped.loc[
            grouped["filing_count"] >= 10
        ].reset_index(drop=True)
        tracker.record("filter", len(grouped), qualifying)

        ordered = qualifying.sort_values(
            ["avg_risk_factor_count", "industry"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(qualifying), ordered)

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
