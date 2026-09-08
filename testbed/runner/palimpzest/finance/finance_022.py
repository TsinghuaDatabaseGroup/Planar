#!/usr/bin/env python3
"""Palimpzest pipeline for finance-022."""

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

TASK_ID = "finance-022"
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
            filings["year"].isin([2021, 2022, 2023])
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

        incorporation_plan = memory_dataset(
            f"{TASK_ID}-incorporation", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it explicitly identifies the reporting "
                "company's legal jurisdiction of incorporation as one of the 50 "
                "U.S. states or the District of Columbia."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        incorporation_result = incorporation_plan.run(config)
        incorporated = result_frame(incorporation_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(incorporated),
            incorporation_result,
            time.time() - started,
        )

        headquarters_plan = memory_dataset(
            f"{TASK_ID}-headquarters", incorporated
        ).sem_filter(
            filter=(
                "Keep the filing only if it identifies the reporting company's "
                "headquarters or principal executive offices as located in a country "
                "outside the United States. Do not infer headquarters from the "
                "incorporation jurisdiction or operating footprint."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        headquarters_result = headquarters_plan.run(config)
        foreign_headquarters = result_frame(headquarters_result, incorporated)
        tracker.record_semantic(
            "sem_filter",
            len(incorporated),
            _intermediate_view(foreign_headquarters),
            headquarters_result,
            time.time() - started,
        )

        answer = int(len(foreign_headquarters))
        tracker.record("groupby", len(foreign_headquarters), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
