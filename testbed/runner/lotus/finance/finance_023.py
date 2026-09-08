#!/usr/bin/env python3
"""LOTUS pipeline for finance-023."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    iter_selected_texts,
    load_table,
    save_output,
    sem_filter_in_batches,
    setup,
)

TASK_ID = "finance-023"
BATCH_SIZE = 100
AVAILABLE_YEARS = tuple(range(2019, 2025))


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _intermediate_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["document_text"], errors="ignore")


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = pd.concat(
            [
                load_table("SEC", f"CSV/{year}.csv")[
                    ["year", "text", "word_count"]
                ].copy()
                for year in AVAILABLE_YEARS
            ],
            ignore_index=True,
        )
        filings["year"] = pd.to_numeric(filings["year"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("SCAN_TABLE(CSV)", None, len(filings), output=filings)

        candidates = filings[
            filings["year"].isin([2022, 2023])
            & (filings["word_count"] > 80000)
        ].copy()
        tracker.record(
            "FILTER(year IN [2022, 2023] AND word_count > 80000)",
            len(filings),
            len(candidates),
            output=candidates,
        )
        tracker.record(
            "SCAN_DOCS(selector=filtered.text)",
            len(candidates),
            len(candidates),
            output=candidates,
        )

        with tracker.step(
            "SEM_FILTER(AI or machine-learning use in company business)",
            input_rows=len(candidates),
        ) as step:
            ai_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                ai_parts.append(
                    documents.sem_filter(
                        "The 10-K filing {document_text} describes the reporting "
                        "company using artificial intelligence or machine learning "
                        "in its business, products, services, operations, or strategy."
                    )
                )
            ai_filings = (
                pd.concat(ai_parts, ignore_index=True)
                if ai_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(ai_filings))

        with tracker.step(
            "SEM_FILTER(no ESG or sustainability topics)",
            input_rows=len(ai_filings),
        ) as step:
            no_esg = sem_filter_in_batches(
                ai_filings,
                "The 10-K filing {document_text} does not mention environmental, "
                "social, governance, ESG, or sustainability topics.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(no_esg))

        answer = int(len(no_esg))
        tracker.record(
            "GROUP_BY([], COUNT(*) AS count)",
            len(no_esg),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
