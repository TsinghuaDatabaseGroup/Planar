#!/usr/bin/env python3
"""LOTUS pipeline for finance-008."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    iter_selected_texts,
    load_table,
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "finance-008"
BATCH_SIZE = 100
AVAILABLE_YEARS = tuple(range(2019, 2025))


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = pd.concat(
            [
                load_table("SEC", f"CSV/{year}.csv")[["year", "text"]].copy()
                for year in AVAILABLE_YEARS
            ],
            ignore_index=True,
        )
        filings["year"] = pd.to_numeric(filings["year"], errors="coerce").astype(
            "Int64"
        )
        tracker.record("SCAN_TABLE(CSV)", None, len(filings), output=filings)

        selected_years = filings[filings["year"].between(2019, 2023)].copy()
        tracker.record(
            "FILTER(year BETWEEN 2019 AND 2023)",
            len(filings),
            len(selected_years),
            output=selected_years,
        )
        tracker.record(
            "SCAN_DOCS(selector=filtered.text)",
            len(selected_years),
            len(selected_years),
            output=selected_years,
        )

        with tracker.step(
            "SEM_EXTRACT(going-concern doubt)",
            input_rows=len(selected_years),
        ) as step:
            extracted_parts = []
            for documents in iter_selected_texts(
                "SEC",
                selected_years,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["document_text"],
                    output_cols={
                        "has_going_concern": (
                            "true only when the filing expresses substantial doubt "
                            "about the reporting company's ability to continue as a "
                            "going concern; false otherwise"
                        )
                    },
                )
                batch["has_going_concern"] = batch["has_going_concern"].map(
                    parse_bool
                )
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else selected_years.iloc[0:0].assign(
                    has_going_concern=pd.Series(dtype="bool")
                )
            )
            step.set_output(extracted)

        answer_frame = (
            extracted.groupby("year", sort=False)
            .agg(
                filing_count=("year", "size"),
                going_concern_filings=("has_going_concern", "sum"),
            )
            .reset_index()
        )
        answer_frame["going_concern_rate_percent"] = (
            100.0
            * answer_frame["going_concern_filings"]
            / answer_frame["filing_count"]
        ).round(1)
        answer_frame = answer_frame[
            ["year", "going_concern_rate_percent"]
        ].sort_values("year", ascending=True, kind="mergesort").reset_index(drop=True)
        tracker.record(
            "GROUP_BY(year, rounded going-concern rate) -> ORDER_BY(year ASC)",
            len(extracted),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} yearly rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
