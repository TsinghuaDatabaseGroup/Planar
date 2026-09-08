#!/usr/bin/env python3
"""LOTUS pipeline for finance-007."""

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

TASK_ID = "finance-007"
BATCH_SIZE = 100
YEARS = tuple(range(2019, 2025))


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = pd.concat(
            [
                load_table("SEC", f"CSV/{year}.csv")[["year", "text"]].copy()
                for year in YEARS
            ],
            ignore_index=True,
        )
        filings["year"] = pd.to_numeric(filings["year"], errors="coerce").astype(
            "Int64"
        )
        tracker.record("SCAN_TABLE(CSV)", None, len(filings), output=filings)
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_EXTRACT(AI or machine-learning business mention)",
            input_rows=len(filings),
        ) as step:
            extracted_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["document_text"],
                    output_cols={
                        "ai_ml_mentioned": (
                            "true only when the filing specifically mentions artificial "
                            "intelligence, machine learning, deep learning, neural "
                            "networks, AI-powered systems, or a similar AI/ML technology "
                            "in the context of the reporting company's business, "
                            "products, or strategy; false otherwise"
                        )
                    },
                )
                batch["ai_ml_mentioned"] = batch["ai_ml_mentioned"].map(parse_bool)
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else filings.iloc[0:0].assign(
                    ai_ml_mentioned=pd.Series(dtype="bool")
                )
            )
            step.set_output(extracted)

        answer_frame = (
            extracted.groupby("year", sort=False)
            .agg(
                filing_count=("year", "size"),
                ai_ml_mentions=("ai_ml_mentioned", "sum"),
            )
            .reset_index()
        )
        answer_frame["ai_ml_mention_rate_percent"] = (
            100.0
            * answer_frame["ai_ml_mentions"]
            / answer_frame["filing_count"]
        ).round(1)
        answer_frame = answer_frame[
            ["year", "ai_ml_mention_rate_percent"]
        ].sort_values("year", ascending=True, kind="mergesort").reset_index(drop=True)
        tracker.record(
            "GROUP_BY(year, rounded AI/ML mention rate) -> ORDER_BY(year ASC)",
            len(extracted),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} yearly rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
