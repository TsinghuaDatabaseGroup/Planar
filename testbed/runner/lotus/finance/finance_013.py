#!/usr/bin/env python3
"""LOTUS pipeline for finance-013."""

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

TASK_ID = "finance-013"
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
            filings["year"].isin([2019, 2020, 2021, 2022, 2023])
            & (filings["word_count"] > 95000)
        ].copy()
        tracker.record(
            "FILTER(year IN [2019, 2020, 2021, 2022, 2023] AND word_count > 95000)",
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
            "SEM_FILTER(explicitly reports exactly zero employees)",
            input_rows=len(candidates),
        ) as step:
            zero_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                selected = documents.sem_filter(
                    "The 10-K filing {document_text} explicitly reports that the "
                    "reporting company has a company-wide total of exactly zero "
                    "employees. A missing, undisclosed, or partial workforce count "
                    "does not qualify."
                )
                zero_parts.append(
                    selected.drop(columns=["document_text"], errors="ignore")
                )
            zero_employee_filings = (
                pd.concat(zero_parts, ignore_index=True)
                if zero_parts
                else candidates.iloc[0:0].copy()
            )
            step.set_output(zero_employee_filings)

        answer = int(len(zero_employee_filings))
        tracker.record(
            "GROUP_BY([], COUNT(*) AS count)",
            len(zero_employee_filings),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
