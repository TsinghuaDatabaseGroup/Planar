#!/usr/bin/env python3
"""LOTUS pipeline for finance-022."""

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

TASK_ID = "finance-022"
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
            filings["year"].isin([2021, 2022, 2023])
            & (filings["word_count"] > 90000)
        ].copy()
        tracker.record(
            "FILTER(year IN [2021, 2022, 2023] AND word_count > 90000)",
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
            "SEM_FILTER(incorporated in a U.S. state or District of Columbia)",
            input_rows=len(candidates),
        ) as step:
            incorporated_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                incorporated_parts.append(
                    documents.sem_filter(
                        "The 10-K filing {document_text} explicitly identifies the "
                        "reporting company's legal jurisdiction of incorporation as "
                        "one of the 50 U.S. states or the District of Columbia."
                    )
                )
            incorporated = (
                pd.concat(incorporated_parts, ignore_index=True)
                if incorporated_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(incorporated))

        with tracker.step(
            "SEM_FILTER(headquarters outside the United States)",
            input_rows=len(incorporated),
        ) as step:
            foreign_headquarters = sem_filter_in_batches(
                incorporated,
                "The 10-K filing {document_text} identifies the reporting company's "
                "headquarters or principal executive offices as located in a country "
                "outside the United States; do not infer headquarters from the "
                "incorporation jurisdiction or operating footprint.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(foreign_headquarters))

        answer = int(len(foreign_headquarters))
        tracker.record(
            "GROUP_BY([], COUNT(*) AS count)",
            len(foreign_headquarters),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
