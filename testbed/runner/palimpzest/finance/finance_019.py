#!/usr/bin/env python3
"""Palimpzest pipeline for finance-019."""

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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "finance-019"
DATASET = "SEC"
YEARS = tuple(range(2019, 2025))
COMPETITIVE_POSITIONS = (
    "market_leader",
    "major_player",
    "challenger",
    "niche_player",
    "emerging",
)


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_text", "primary_2024_filing"],
        errors="ignore",
    )


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        annual_frames = [
            load_table(
                DATASET,
                f"CSV/{year}.csv",
                ["cik", "year", "text", "word_count"],
            )
            for year in YEARS
        ]
        filings = pd.concat(annual_frames, ignore_index=True)
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["year"] = pd.to_numeric(
            filings["year"], errors="coerce"
        ).astype("Int64")
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["cik", "year", "word_count", "document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        classification_plan = memory_dataset(
            f"{TASK_ID}-competitive-position", documents
        ).sem_map(
            cols=[
                {
                    "name": "competitive_position",
                    "type": str,
                    "desc": (
                        "Exactly one category: market_leader, major_player, "
                        "challenger, niche_player, or emerging."
                    ),
                }
            ],
            desc=(
                "Assign the filing to one competitive-position category for the "
                "reporting company."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            documents,
            ["competitive_position"],
        )
        classified["competitive_position"] = classified[
            "competitive_position"
        ].map(lambda value: normalize_enum(value, COMPETITIVE_POSITIONS))
        classified["is_primary_2024_candidate"] = (
            classified["year"].eq(2024)
            & ~classified["document_text"].str.contains(
                "FORM 10-K/A",
                case=False,
                regex=False,
                na=False,
            )
        )
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(classified),
            classification_result,
            time.time() - started,
        )

        leader_years = (
            classified.loc[
                classified["competitive_position"].eq("market_leader")
            ]
            .groupby("cik")["year"]
            .nunique()
            .rename("leader_years")
        )
        primary_2024 = (
            classified.loc[classified["is_primary_2024_candidate"]]
            .sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .set_index("cik")["document_text"]
            .rename("primary_2024_filing")
        )
        grouped = pd.DataFrame(
            {"cik": classified["cik"].dropna().drop_duplicates()}
        ).reset_index(drop=True)
        grouped = grouped.join(leader_years, on="cik").join(
            primary_2024, on="cik"
        )
        grouped["leader_years"] = grouped["leader_years"].fillna(0).astype(int)
        tracker.record("groupby", len(classified), _intermediate_view(grouped))

        six_year_leaders = grouped.loc[grouped["leader_years"].eq(6)].reset_index(
            drop=True
        )
        tracker.record(
            "filter",
            len(grouped),
            _intermediate_view(six_year_leaders),
        )

        available = six_year_leaders.loc[
            six_year_leaders["primary_2024_filing"].notna()
        ].reset_index(drop=True)
        extraction_plan = memory_dataset(
            f"{TASK_ID}-company-2024", available
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": (
                        "The reporting company's legal name stated in the retained "
                        "non-amended 2024 filing."
                    ),
                },
                {
                    "name": "industry",
                    "type": str,
                    "desc": (
                        "The reporting company's industry sector stated or clearly "
                        "supported by the retained non-amended 2024 filing."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name and industry sector from the "
                "retained non-amended 2024 filing."
            ),
            depends_on=["primary_2024_filing"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            available,
            ["company_name", "industry"],
        )
        missing = six_year_leaders.loc[
            six_year_leaders["primary_2024_filing"].isna()
        ].copy()
        if not missing.empty:
            missing["company_name"] = None
            missing["industry"] = None
            extracted = pd.concat([extracted, missing], ignore_index=True)
        tracker.record_semantic(
            "sem_map",
            len(six_year_leaders),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        answer_frame = extracted[["company_name", "industry"]].sort_values(
            "company_name",
            ascending=True,
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
        tracker.record("order_by", len(extracted), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
