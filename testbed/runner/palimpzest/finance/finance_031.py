#!/usr/bin/env python3
"""Palimpzest pipeline for finance-031."""

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

TASK_ID = "finance-031"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_2019", "document_2020"],
        errors="ignore",
    )


def _mean_change(adopters: pd.DataFrame) -> float | None:
    complete = adopters.dropna(
        subset=["risk_factor_count_2019", "risk_factor_count_2020"]
    )
    if complete.empty:
        return None
    total = sum(
        Decimal(str(current)) - Decimal(str(previous))
        for previous, current in zip(
            complete["risk_factor_count_2019"],
            complete["risk_factor_count_2020"],
        )
    )
    return float(
        (total / Decimal(len(complete))).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
    )


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

        filings_2020 = load_table(
            DATASET,
            "CSV/2020.csv",
            ["cik", "text"],
        )
        filings_2020["cik"] = pd.to_numeric(
            filings_2020["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2020)

        documents_2020 = load_selected_texts(
            DATASET,
            filings_2020,
            path_column="text",
            output_column="document_2020",
        )[["cik", "document_2020"]]
        tracker.record(
            "scan",
            len(filings_2020),
            _intermediate_view(documents_2020),
        )

        paired = documents_2019.loc[documents_2019["cik"].notna()].merge(
            documents_2020.loc[documents_2020["cik"].notna()],
            on="cik",
            how="inner",
            sort=False,
        )
        tracker.record(
            "join",
            {"left": len(documents_2019), "right": len(documents_2020)},
            _intermediate_view(paired),
        )

        pandemic_plan = memory_dataset(
            f"{TASK_ID}-pandemic-2020", paired
        ).sem_filter(
            filter=(
                "Keep the filing pair only if the 2020 filing includes a pandemic "
                "or public-health risk factor."
            ),
            depends_on=["document_2020"],
        )
        started = time.time()
        pandemic_result = pandemic_plan.run(config)
        pandemic_2020 = result_frame(pandemic_result, paired)
        tracker.record_semantic(
            "sem_filter",
            len(paired),
            _intermediate_view(pandemic_2020),
            pandemic_result,
            time.time() - started,
        )

        new_adopter_plan = memory_dataset(
            f"{TASK_ID}-new-adopters", pandemic_2020
        ).sem_filter(
            filter=(
                "Keep the filing pair only if the 2019 filing does not include a "
                "pandemic or public-health risk factor."
            ),
            depends_on=["document_2019"],
        )
        started = time.time()
        new_adopter_result = new_adopter_plan.run(config)
        new_adopters = result_frame(new_adopter_result, pandemic_2020)
        tracker.record_semantic(
            "sem_filter",
            len(pandemic_2020),
            _intermediate_view(new_adopters),
            new_adopter_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-annual-details", new_adopters
        ).sem_map(
            cols=[
                {
                    "name": "risk_factor_count_2019",
                    "type": int,
                    "desc": (
                        "The total number of distinct risk factors in the 2019 "
                        "filing's Risk Factors section, or null when no total can be "
                        "determined."
                    ),
                },
                {
                    "name": "risk_factor_count_2020",
                    "type": int,
                    "desc": (
                        "The total number of distinct risk factors in the 2020 "
                        "filing's Risk Factors section, or null when no total can be "
                        "determined."
                    ),
                },
                {
                    "name": "industry_sector_2020",
                    "type": str,
                    "desc": (
                        "The reporting company's concise industry sector in the "
                        "2020 filing."
                    ),
                },
            ],
            desc=(
                "Extract the total risk-factor count from each annual filing and "
                "the 2020 industry sector."
            ),
            depends_on=["document_2019", "document_2020"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        adopters = result_frame(
            extraction_result,
            new_adopters,
            [
                "risk_factor_count_2019",
                "risk_factor_count_2020",
                "industry_sector_2020",
            ],
        )
        for column in ("risk_factor_count_2019", "risk_factor_count_2020"):
            adopters[column] = pd.to_numeric(
                adopters[column], errors="coerce"
            ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(new_adopters),
            _intermediate_view(adopters),
            extraction_result,
            time.time() - started,
        )

        sector_counts = (
            adopters.groupby("industry_sector_2020", sort=False)["cik"]
            .nunique()
            .rename("adopter_count")
            .reset_index()
        )
        tracker.record("groupby", len(adopters), sector_counts)

        ordered_sectors = sector_counts.sort_values(
            ["adopter_count", "industry_sector_2020"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(sector_counts), ordered_sectors)

        top_sectors = ordered_sectors.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered_sectors), top_sectors)

        mean_change = _mean_change(adopters)
        change_summary = {"mean_risk_factor_count_change": mean_change}
        tracker.record("groupby", len(adopters), change_summary)

        answer = {
            "top_sectors": df_records(top_sectors),
            "mean_risk_factor_count_change": mean_change,
        }
        tracker.record(
            "project",
            {"top_sectors": len(top_sectors), "change_summary": 1},
            answer,
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
