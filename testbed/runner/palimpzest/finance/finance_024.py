#!/usr/bin/env python3
"""Palimpzest pipeline for finance-024."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "finance-024"
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
                    ["year", "text", "word_count"],
                )
                for year in AVAILABLE_YEARS
            ],
            ignore_index=True,
        )
        filings["year"] = pd.to_numeric(
            filings["year"], errors="coerce"
        ).astype("Int64")
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[
            filings["year"].isin([2019, 2020, 2021])
            & filings["word_count"].gt(80_000)
        ].reset_index(drop=True)
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["year", "word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        international_plan = memory_dataset(
            f"{TASK_ID}-international", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it explicitly states that the reporting "
                "company has actual operations outside the United States. Do not "
                "treat foreign customers, foreign revenue, suppliers, investments, "
                "or a generic ability to operate internationally as actual "
                "international operations."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        international_result = international_plan.run(config)
        international = result_frame(international_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(international),
            international_result,
            time.time() - started,
        )

        no_currency_plan = memory_dataset(
            f"{TASK_ID}-no-currency-risk", international
        ).sem_filter(
            filter=(
                "Keep the filing only if it does not disclose foreign-currency or "
                "exchange-rate risk for the reporting company. A disclosure that "
                "the exposure is immaterial or hedged still counts as a disclosure "
                "and therefore does not qualify."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        no_currency_result = no_currency_plan.run(config)
        no_currency_risk = result_frame(no_currency_result, international)
        tracker.record_semantic(
            "sem_filter",
            len(international),
            _intermediate_view(no_currency_risk),
            no_currency_result,
            time.time() - started,
        )

        answer = int(len(no_currency_risk))
        tracker.record("groupby", len(no_currency_risk), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
