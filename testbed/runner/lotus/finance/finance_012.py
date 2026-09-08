#!/usr/bin/env python3
"""LOTUS pipeline for finance-012."""

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
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "finance-012"
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

        selected_years = filings[filings["year"].isin([2022, 2023])].copy()
        tracker.record(
            "FILTER(year IN [2022, 2023])",
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
            "SEM_EXTRACT(industry sector and AI/ML mention)",
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
                        "industry_sector": (
                            "the reporting company's concise industry sector stated "
                            "or clearly supported by the filing"
                        ),
                        "ai_ml_mentioned": (
                            "true when the filing mentions artificial-intelligence or "
                            "machine-learning technologies; false otherwise"
                        ),
                    },
                )
                batch["industry_sector"] = batch["industry_sector"].map(clean_text)
                batch["ai_ml_mentioned"] = batch["ai_ml_mentioned"].map(parse_bool)
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else selected_years.iloc[0:0].assign(
                    industry_sector=pd.Series(dtype="object"),
                    ai_ml_mentioned=pd.Series(dtype="bool"),
                )
            )
            step.set_output(extracted)

        grouped_rows = []
        for industry_sector, group in extracted.groupby(
            "industry_sector", sort=False
        ):
            is_2022 = group["year"] == 2022
            is_2023 = group["year"] == 2023
            grouped_rows.append(
                {
                    "industry_sector": industry_sector,
                    "ai_ml_filings_2022": int(
                        (is_2022 & group["ai_ml_mentioned"]).sum()
                    ),
                    "ai_ml_filings_2023": int(
                        (is_2023 & group["ai_ml_mentioned"]).sum()
                    ),
                    "total_filings_2023": int(is_2023.sum()),
                }
            )
        grouped = pd.DataFrame(
            grouped_rows,
            columns=[
                "industry_sector",
                "ai_ml_filings_2022",
                "ai_ml_filings_2023",
                "total_filings_2023",
            ],
        )
        tracker.record(
            "GROUP_BY(industry_sector, conditional 2022/2023 AI counts and 2023 total)",
            len(extracted),
            len(grouped),
            output=grouped,
        )

        qualifying = grouped[
            (grouped["total_filings_2023"] >= 20)
            & (grouped["ai_ml_filings_2023"] > grouped["ai_ml_filings_2022"])
        ].copy()
        qualifying["_increase"] = (
            qualifying["ai_ml_filings_2023"]
            - qualifying["ai_ml_filings_2022"]
        )
        tracker.record(
            "FILTER(total_filings_2023 >= 20 AND ai_ml_filings_2023 > ai_ml_filings_2022)",
            len(grouped),
            len(qualifying),
            output=qualifying.drop(columns=["_increase"]),
        )

        answer_frame = qualifying.sort_values(
            ["_increase", "industry_sector"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10).drop(columns=["_increase"]).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(AI/ML increase DESC, industry_sector ASC) -> LIMIT(10)",
            len(qualifying),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
