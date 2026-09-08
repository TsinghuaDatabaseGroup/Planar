#!/usr/bin/env python3
"""Palimpzest pipeline for finance-029."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "finance-029"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings_2019 = load_table(
            DATASET,
            "CSV/2019.csv",
            ["cik", "text"],
        )
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2019)

        documents_2019 = load_selected_texts(
            DATASET,
            filings_2019,
            path_column="text",
            output_column="document_text",
        )
        tracker.record(
            "scan",
            len(filings_2019),
            _intermediate_view(documents_2019),
        )

        challenger_plan = memory_dataset(
            f"{TASK_ID}-challengers-2019", documents_2019
        ).sem_filter(
            filter=(
                "Keep the filing only if it describes the reporting company as "
                "holding a challenger competitive position in its industry or "
                "primary market in 2019."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        challenger_result = challenger_plan.run(config)
        challenger_filings = result_frame(challenger_result, documents_2019)
        tracker.record_semantic(
            "sem_filter",
            len(documents_2019),
            _intermediate_view(challenger_filings),
            challenger_result,
            time.time() - started,
        )

        challengers_2019 = challenger_filings[["cik"]].reset_index(drop=True)
        tracker.record("project", len(challenger_filings), challengers_2019)

        filings_2023 = load_table(
            DATASET,
            "CSV/2023.csv",
            ["cik", "text"],
        )
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2023)

        documents_2023 = load_selected_texts(
            DATASET,
            filings_2023,
            path_column="text",
            output_column="document_text",
        )
        tracker.record(
            "scan",
            len(filings_2023),
            _intermediate_view(documents_2023),
        )

        leader_plan = memory_dataset(
            f"{TASK_ID}-leaders-2023", documents_2023
        ).sem_filter(
            filter=(
                "Keep the filing only if it describes the reporting company as a "
                "recognized industry or market leader in 2023."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        leader_result = leader_plan.run(config)
        leader_filings = result_frame(leader_result, documents_2023)
        tracker.record_semantic(
            "sem_filter",
            len(documents_2023),
            _intermediate_view(leader_filings),
            leader_result,
            time.time() - started,
        )

        deduped_leaders = leader_filings.drop_duplicates(
            "cik", keep="first"
        ).reset_index(drop=True)
        tracker.record(
            "dedup",
            len(leader_filings),
            _intermediate_view(deduped_leaders),
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-leader-details", deduped_leaders
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The reporting company's legal name.",
                },
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": (
                        "The reporting company's industry sector stated or clearly "
                        "supported by the filing."
                    ),
                },
            ],
            desc="Extract the company legal name and industry sector.",
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted_leaders = result_frame(
            extraction_result,
            deduped_leaders,
            ["company_name", "industry_sector"],
        )
        tracker.record_semantic(
            "sem_map",
            len(deduped_leaders),
            _intermediate_view(extracted_leaders),
            extraction_result,
            time.time() - started,
        )

        matched = challengers_2019.loc[
            challengers_2019["cik"].notna()
        ].merge(
            extracted_leaders.loc[extracted_leaders["cik"].notna()],
            on="cik",
            how="inner",
            sort=False,
        )
        tracker.record(
            "join",
            {
                "left": len(challengers_2019),
                "right": len(extracted_leaders),
            },
            _intermediate_view(matched),
        )

        answer_frame = matched[["company_name", "industry_sector"]].reset_index(
            drop=True
        )
        tracker.record("project", len(matched), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
