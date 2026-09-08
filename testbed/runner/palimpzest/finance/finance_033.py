#!/usr/bin/env python3
"""Palimpzest pipeline for finance-033."""

from __future__ import annotations

import os
import sys
import time
from decimal import ROUND_HALF_UP, Decimal

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

TASK_ID = "finance-033"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_2019", "document_2023"],
        errors="ignore",
    )


def _employee_change_percent(previous, current) -> float | None:
    if pd.isna(previous) or pd.isna(current) or int(previous) == 0:
        return None
    change = (
        Decimal(100)
        * (Decimal(int(current)) - Decimal(int(previous)))
        / Decimal(int(previous))
    )
    return float(change.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings_2019 = load_table(
            DATASET,
            "CSV/2019.csv",
            ["cik", "text"],
        )
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2019)

        documents_2019 = load_selected_texts(
            DATASET,
            filings_2019,
            path_column="text",
            output_column="document_2019",
        )[["cik", "document_2019"]]
        tracker.record(
            "scan",
            len(filings_2019),
            _intermediate_view(documents_2019),
        )

        extraction_2019_plan = memory_dataset(
            f"{TASK_ID}-employees-2019", documents_2019
        ).sem_map(
            cols=[
                {
                    "name": "employee_count_2019",
                    "type": int,
                    "desc": (
                        "Only an explicitly disclosed company-wide total employee "
                        "count; null for partial workforce figures, constructed "
                        "totals, or non-headcount numbers."
                    ),
                }
            ],
            desc="Extract the explicitly disclosed company-wide employee count.",
            depends_on=["document_2019"],
        )
        started = time.time()
        extraction_2019_result = extraction_2019_plan.run(config)
        employees_2019 = result_frame(
            extraction_2019_result,
            documents_2019,
            ["employee_count_2019"],
        )
        employees_2019["employee_count_2019"] = pd.to_numeric(
            employees_2019["employee_count_2019"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(documents_2019),
            _intermediate_view(employees_2019),
            extraction_2019_result,
            time.time() - started,
        )

        eligible_2019 = employees_2019.loc[
            employees_2019["employee_count_2019"] >= 1_000
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(employees_2019),
            _intermediate_view(eligible_2019),
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
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(filings_2023), primary_2023)

        documents_2023 = load_selected_texts(
            DATASET,
            primary_2023,
            path_column="primary_text_2023",
            output_column="document_2023",
        )[["cik", "word_count", "document_2023"]]
        tracker.record(
            "scan",
            len(primary_2023),
            _intermediate_view(documents_2023),
        )

        extraction_2023_plan = memory_dataset(
            f"{TASK_ID}-employees-2023", documents_2023
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The reporting company's legal name.",
                },
                {
                    "name": "employee_count_2023",
                    "type": int,
                    "desc": (
                        "Only an explicitly disclosed company-wide total employee "
                        "count; null for partial workforce figures, constructed "
                        "totals, or non-headcount numbers."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name and explicitly disclosed "
                "company-wide employee count."
            ),
            depends_on=["document_2023"],
        )
        started = time.time()
        extraction_2023_result = extraction_2023_plan.run(config)
        employees_2023 = result_frame(
            extraction_2023_result,
            documents_2023,
            ["company_name", "employee_count_2023"],
        )
        employees_2023["employee_count_2023"] = pd.to_numeric(
            employees_2023["employee_count_2023"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(documents_2023),
            _intermediate_view(employees_2023),
            extraction_2023_result,
            time.time() - started,
        )

        paired = eligible_2019.loc[eligible_2019["cik"].notna()].merge(
            employees_2023.loc[employees_2023["cik"].notna()],
            on="cik",
            how="inner",
            sort=False,
        )
        tracker.record(
            "join",
            {"left": len(eligible_2019), "right": len(employees_2023)},
            _intermediate_view(paired),
        )

        changes = paired[
            ["company_name", "employee_count_2019", "employee_count_2023"]
        ].copy()
        changes["employee_change_percent"] = [
            _employee_change_percent(previous, current)
            for previous, current in zip(
                changes["employee_count_2019"],
                changes["employee_count_2023"],
            )
        ]
        tracker.record("project", len(paired), changes)

        positive = changes.loc[
            changes["employee_change_percent"].notna()
            & changes["employee_change_percent"].gt(0)
        ].reset_index(drop=True)
        tracker.record("filter", len(changes), positive)

        ordered_growth = positive.sort_values(
            ["employee_change_percent", "company_name"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(positive), ordered_growth)

        top_growth = ordered_growth.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered_growth), top_growth)

        negative = changes.loc[
            changes["employee_change_percent"].notna()
            & changes["employee_change_percent"].lt(0)
        ].reset_index(drop=True)
        tracker.record("filter", len(changes), negative)

        ordered_decline = negative.sort_values(
            ["employee_change_percent", "company_name"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(negative), ordered_decline)

        top_decline = ordered_decline.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered_decline), top_decline)

        answer = {
            "top_5_growth": df_records(top_growth),
            "top_5_decline": df_records(top_decline),
        }
        tracker.record(
            "project",
            {"growth": len(top_growth), "decline": len(top_decline)},
            answer,
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
