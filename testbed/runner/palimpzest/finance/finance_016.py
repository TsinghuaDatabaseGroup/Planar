#!/usr/bin/env python3
"""Palimpzest pipeline for finance-016."""

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

TASK_ID = "finance-016"
DATASET = "SEC"


def _mentions_ai_or_ml(texts: pd.Series) -> pd.Series:
    return texts.str.contains(
        "artificial intelligence", case=False, regex=False, na=False
    ) | texts.str.contains("machine learning", case=False, regex=False, na=False)


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_2019", "document_2023"],
        errors="ignore",
    )


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings_2019 = load_table(
            DATASET,
            "CSV/2019.csv",
            ["id", "cik", "text"],
        ).rename(columns={"id": "id_2019", "text": "text_2019"})
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2019)

        filings_2023 = load_table(
            DATASET,
            "CSV/2023.csv",
            ["id", "cik", "text"],
        ).rename(columns={"id": "id_2023", "text": "text_2023"})
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2023)

        joined = filings_2019.loc[filings_2019["cik"].notna()].merge(
            filings_2023.loc[filings_2023["cik"].notna()],
            on="cik",
            how="inner",
            sort=False,
        )
        tracker.record(
            "join",
            {"left": len(filings_2019), "right": len(filings_2023)},
            joined,
        )

        documents_2023 = load_selected_texts(
            DATASET,
            joined,
            path_column="text_2023",
            output_column="document_2023",
        )
        tracker.record(
            "scan",
            len(joined),
            _intermediate_view(documents_2023),
        )

        mentions_2023 = documents_2023.loc[
            _mentions_ai_or_ml(documents_2023["document_2023"])
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(documents_2023),
            _intermediate_view(mentions_2023),
        )

        deduped = mentions_2023.drop_duplicates(
            "cik", keep="first"
        ).reset_index(drop=True)
        tracker.record("dedup", len(mentions_2023), _intermediate_view(deduped))

        documents_2019 = load_selected_texts(
            DATASET,
            deduped,
            path_column="text_2019",
            output_column="document_2019",
        )
        tracker.record(
            "scan",
            len(deduped),
            _intermediate_view(documents_2019),
        )

        no_mentions_2019 = documents_2019.loc[
            ~_mentions_ai_or_ml(documents_2019["document_2019"])
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(documents_2019),
            _intermediate_view(no_mentions_2019),
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-industry", no_mentions_2019
        ).sem_map(
            cols=[
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": "The company's primary industry sector in 2023.",
                }
            ],
            desc="Extract the 2023 industry sector.",
            depends_on=["document_2023"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            no_mentions_2019,
            ["industry_sector"],
        )
        tracker.record_semantic(
            "sem_map",
            len(no_mentions_2019),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("industry_sector")
            .size()
            .rename("company_count")
            .reset_index()
        )
        tracker.record("groupby", len(extracted), grouped)

        ordered = grouped.sort_values(
            ["company_count", "industry_sector"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(grouped), ordered)

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
