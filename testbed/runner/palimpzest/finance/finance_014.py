#!/usr/bin/env python3
"""Palimpzest pipeline for finance-014."""

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

TASK_ID = "finance-014"
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
            filings["year"].isin([2020, 2021, 2022, 2023])
            & filings["word_count"].gt(90_000)
        ].reset_index(drop=True)
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["year", "word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        fiscal_year_plan = memory_dataset(
            f"{TASK_ID}-fiscal-year-end", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it discloses an identifiable fiscal-year-"
                "end date whose normalized month and day are not December 31. Use "
                "the fiscal year end, not the filing date, report date, or another "
                "date; an unavailable fiscal-year-end date does not qualify."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        fiscal_year_result = fiscal_year_plan.run(config)
        non_december = result_frame(fiscal_year_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(non_december),
            fiscal_year_result,
            time.time() - started,
        )

        answer = int(len(non_december))
        tracker.record("groupby", len(non_december), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
