#!/usr/bin/env python3
"""LOTUS pipeline for finance-014."""

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
    setup,
)

TASK_ID = "finance-014"
BATCH_SIZE = 100
AVAILABLE_YEARS = tuple(range(2019, 2025))


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
            filings["year"].isin([2020, 2021, 2022, 2023])
            & (filings["word_count"] > 90000)
        ].copy()
        tracker.record(
            "FILTER(year IN [2020, 2021, 2022, 2023] AND word_count > 90000)",
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
            "SEM_FILTER(identifiable fiscal year end is not December 31)",
            input_rows=len(candidates),
        ) as step:
            non_december_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                selected = documents.sem_filter(
                    "The 10-K filing {document_text} discloses an identifiable fiscal-"
                    "year-end date whose normalized month and day are not December "
                    "31. Use the fiscal year end, not the filing date, report date, "
                    "or another date; an unavailable fiscal-year-end date does not "
                    "qualify."
                )
                non_december_parts.append(
                    selected.drop(columns=["document_text"], errors="ignore")
                )
            non_december = (
                pd.concat(non_december_parts, ignore_index=True)
                if non_december_parts
                else candidates.iloc[0:0].copy()
            )
            step.set_output(non_december)

        answer = int(len(non_december))
        tracker.record(
            "GROUP_BY([], COUNT(*) AS count)",
            len(non_december),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
