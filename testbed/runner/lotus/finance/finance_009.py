#!/usr/bin/env python3
"""LOTUS pipeline for finance-009."""

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
    normalize_enum,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "finance-009"
BATCH_SIZE = 100
COMPETITIVE_POSITIONS = ("market_leader", "other")


def _capped_advantage_count(value) -> int:
    parsed = parse_number(value)
    if parsed is None:
        return 0
    return min(max(int(parsed), 0), 8)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2023.csv")[["text"]].copy()
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv)", None, len(filings), output=filings
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_EXTRACT(capped advantages and market-leader classification)",
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
                        "competitive_advantage_count": (
                            "the number of distinct competitive-advantage themes "
                            "described for the reporting company in the filing, "
                            "capped at eight; return an integer from 0 through 8"
                        ),
                        "competitive_position": (
                            "market_leader when the filing describes the reporting "
                            "company as a market leader, or other otherwise"
                        ),
                    },
                )
                batch["competitive_advantage_count"] = batch[
                    "competitive_advantage_count"
                ].map(_capped_advantage_count)
                batch["competitive_position"] = batch[
                    "competitive_position"
                ].map(
                    lambda value: normalize_enum(value, COMPETITIVE_POSITIONS)
                )
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else filings.iloc[0:0].assign(
                    competitive_advantage_count=pd.Series(dtype="int64"),
                    competitive_position=pd.Series(dtype="object"),
                )
            )
            step.set_output(extracted)

        projected = pd.DataFrame(
            {
                "advantage_group": extracted["competitive_advantage_count"].map(
                    lambda value: (
                        "high_advantage_group"
                        if value == 8
                        else "low_advantage_group"
                    )
                ),
                "is_market_leader": (
                    extracted["competitive_position"] == "market_leader"
                ),
            }
        )
        tracker.record(
            "PROJECT(advantage_group, is_market_leader)",
            len(extracted),
            len(projected),
            output=projected,
        )

        answer_frame = (
            projected.groupby("advantage_group", sort=False)
            .agg(
                filing_count=("advantage_group", "size"),
                market_leader_rate_percent=("is_market_leader", "mean"),
            )
            .reset_index()
        )
        answer_frame["market_leader_rate_percent"] = (
            100.0 * answer_frame["market_leader_rate_percent"]
        ).round(1)
        tracker.record(
            "GROUP_BY(advantage_group, filing_count, rounded market-leader rate)",
            len(projected),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} group rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
