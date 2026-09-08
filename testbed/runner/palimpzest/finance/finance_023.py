#!/usr/bin/env python3
"""Palimpzest pipeline for finance-023."""

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

TASK_ID = "finance-023"
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
            filings["year"].isin([2022, 2023])
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

        ai_plan = memory_dataset(f"{TASK_ID}-ai-use", documents).sem_filter(
            filter=(
                "Keep the filing only if it describes the reporting company using "
                "artificial intelligence or machine learning in its business, "
                "products, services, operations, or strategy."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        ai_result = ai_plan.run(config)
        ai_filings = result_frame(ai_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(ai_filings),
            ai_result,
            time.time() - started,
        )

        no_esg_plan = memory_dataset(
            f"{TASK_ID}-no-esg", ai_filings
        ).sem_filter(
            filter=(
                "Keep the filing only if it does not mention environmental, social, "
                "governance, ESG, or sustainability topics."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        no_esg_result = no_esg_plan.run(config)
        no_esg = result_frame(no_esg_result, ai_filings)
        tracker.record_semantic(
            "sem_filter",
            len(ai_filings),
            _intermediate_view(no_esg),
            no_esg_result,
            time.time() - started,
        )

        answer = int(len(no_esg))
        tracker.record("groupby", len(no_esg), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
