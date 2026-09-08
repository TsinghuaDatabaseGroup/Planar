#!/usr/bin/env python3
"""LOTUS pipeline for finance-019."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    iter_selected_texts,
    load_table,
    normalize_enum,
    save_output,
    setup,
)

TASK_ID = "finance-019"
BATCH_SIZE = 100
YEARS = tuple(range(2019, 2025))
COMPETITIVE_POSITIONS = (
    "market_leader",
    "major_player",
    "challenger",
    "niche_player",
    "emerging",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        annual_frames = []
        for year in YEARS:
            annual_frames.append(
                load_table("SEC", f"CSV/{year}.csv")[
                    ["cik", "year", "text", "word_count"]
                ].copy()
            )
        filings = pd.concat(annual_frames, ignore_index=True)
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["year"] = pd.to_numeric(filings["year"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("SCAN_TABLE(CSV)", None, len(filings), output=filings)
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_CLASSIFY(competitive-position category)",
            input_rows=len(filings),
        ) as step:
            classified_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_map(
                    "Assign the 10-K filing {document_text} to exactly one "
                    "competitive-position category for the reporting company. "
                    "Output exactly one label: market_leader, major_player, "
                    "challenger, niche_player, or emerging.",
                    suffix="competitive_position",
                )
                batch["competitive_position"] = batch[
                    "competitive_position"
                ].map(lambda value: normalize_enum(value, COMPETITIVE_POSITIONS))
                batch["is_primary_2024_candidate"] = (
                    (batch["year"] == 2024)
                    & ~batch["document_text"].str.contains(
                        "FORM 10-K/A", case=False, regex=False, na=False
                    )
                )
                classified_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            classified = (
                pd.concat(classified_parts, ignore_index=True)
                if classified_parts
                else filings.iloc[0:0].assign(
                    competitive_position=pd.Series(dtype="object"),
                    is_primary_2024_candidate=pd.Series(dtype="bool"),
                )
            )
            step.set_output(classified)

        leader_years = (
            classified[classified["competitive_position"] == "market_leader"]
            .groupby("cik")["year"]
            .nunique()
            .rename("leader_years")
        )
        primary_2024 = (
            classified[classified["is_primary_2024_candidate"]]
            .sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .set_index("cik")["text"]
            .rename("primary_text_2024")
        )
        grouped = pd.DataFrame(
            {"cik": classified["cik"].dropna().drop_duplicates()}
        ).reset_index(drop=True)
        grouped = grouped.join(leader_years, on="cik").join(primary_2024, on="cik")
        grouped["leader_years"] = grouped["leader_years"].fillna(0).astype(int)
        tracker.record(
            "GROUP_BY(cik, COUNT_DISTINCT(leader year), MAX_BY(non-amended 2024 filing, word_count))",
            len(classified),
            len(grouped),
            output=grouped,
        )

        six_year_leaders = grouped[grouped["leader_years"] == 6].copy()
        tracker.record(
            "FILTER(leader_years = 6)",
            len(grouped),
            len(six_year_leaders),
            output=six_year_leaders,
        )

        with tracker.step(
            "SEM_EXTRACT(company and industry from non-amended 2024 filing)",
            input_rows=len(six_year_leaders),
        ) as step:
            available = six_year_leaders[
                six_year_leaders["primary_text_2024"].notna()
            ].copy()
            extracted_parts = []
            for documents in iter_selected_texts(
                "SEC",
                available,
                path_column="primary_text_2024",
                output_column="primary_2024_filing",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["primary_2024_filing"],
                    output_cols={
                        "company_name": (
                            "the reporting company's legal name stated in the "
                            "retained non-amended 2024 filing"
                        ),
                        "industry": (
                            "the reporting company's industry sector stated or "
                            "clearly supported by the retained non-amended 2024 filing"
                        ),
                    },
                )
                batch["company_name"] = batch["company_name"].map(clean_text)
                batch["industry"] = batch["industry"].map(clean_text)
                extracted_parts.append(
                    batch.drop(columns=["primary_2024_filing"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else available.iloc[0:0].assign(
                    company_name=pd.Series(dtype="object"),
                    industry=pd.Series(dtype="object"),
                )
            )
            missing = six_year_leaders[
                six_year_leaders["primary_text_2024"].isna()
            ].copy()
            if not missing.empty:
                missing["company_name"] = None
                missing["industry"] = None
                extracted = pd.concat([extracted, missing], ignore_index=True)
            step.set_output(extracted)

        answer_frame = extracted[["company_name", "industry"]].sort_values(
            "company_name",
            ascending=True,
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(company_name ASC)",
            len(extracted),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
