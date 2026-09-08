#!/usr/bin/env python3
"""LOTUS pipeline for finance-006."""

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
    parse_number,
    save_output,
    setup,
)

TASK_ID = "finance-006"
BATCH_SIZE = 100


def _capped_count(value, maximum: int) -> int:
    parsed = parse_number(value)
    if parsed is None:
        return 0
    return min(max(int(parsed), 0), maximum)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2022.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2022.csv)", None, len(filings), output=filings
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_EXTRACT(company, industry, capped subsidiaries, and capped advantages)",
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
                        "company_name": (
                            "the reporting company's legal name stated in the filing"
                        ),
                        "industry_sector": (
                            "the reporting company's industry sector stated or "
                            "clearly supported by the filing"
                        ),
                        "subsidiary_count": (
                            "the number of distinct specifically named legal "
                            "subsidiaries disclosed in the filing, excluding the "
                            "registrant itself, capped at ten; return an integer from "
                            "0 through 10"
                        ),
                        "competitive_advantages_count": (
                            "the number of distinct competitive-advantage themes "
                            "described for the reporting company in the filing, "
                            "capped at eight; return an integer from 0 through 8"
                        ),
                    },
                )
                batch["company_name"] = batch["company_name"].map(clean_text)
                batch["industry_sector"] = batch["industry_sector"].map(
                    clean_text
                )
                batch["subsidiary_count"] = batch["subsidiary_count"].map(
                    lambda value: _capped_count(value, 10)
                )
                batch["competitive_advantages_count"] = batch[
                    "competitive_advantages_count"
                ].map(lambda value: _capped_count(value, 8))
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else filings.iloc[0:0].assign(
                    company_name=pd.Series(dtype="object"),
                    industry_sector=pd.Series(dtype="object"),
                    subsidiary_count=pd.Series(dtype="int64"),
                    competitive_advantages_count=pd.Series(dtype="int64"),
                )
            )
            step.set_output(extracted)

        qualifying = extracted[
            (extracted["subsidiary_count"] >= 8)
            & (extracted["competitive_advantages_count"] >= 6)
        ].copy()
        tracker.record(
            "FILTER(subsidiary_count >= 8 AND competitive_advantages_count >= 6)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        qualifying["combined_count"] = (
            qualifying["competitive_advantages_count"]
            + qualifying["subsidiary_count"]
        )
        selected = (
            qualifying.sort_values(
                ["cik", "combined_count", "word_count"],
                ascending=[True, False, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .copy()
        )
        tracker.record(
            "GROUP_BY(cik, MAX_BY(record ORDER BY combined_count DESC, word_count DESC))",
            len(qualifying),
            len(selected),
            output=selected,
        )

        ranked = selected.sort_values(
            ["combined_count", "company_name"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)
        answer_frame = ranked[
            [
                "company_name",
                "competitive_advantages_count",
                "subsidiary_count",
                "industry_sector",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(combined_count DESC, company_name ASC) -> LIMIT(10) -> PROJECT",
            len(selected),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
