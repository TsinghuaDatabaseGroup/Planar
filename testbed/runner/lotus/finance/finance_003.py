#!/usr/bin/env python3
"""LOTUS pipeline for finance-003."""

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

TASK_ID = "finance-003"
BATCH_SIZE = 100


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2023.csv")[["id", "cik", "text"]].copy()
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
            "SEM_EXTRACT(industry sector and distinct Item 1A risk-factor count)",
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
                        "industry": "the company's primary industry sector",
                        "risk_factor_count": (
                            "the number of distinct risk factors listed in Item 1A as "
                            "an integer, or null when the count cannot be determined "
                            "reliably"
                        ),
                    },
                )
                batch["industry"] = batch["industry"].map(clean_text)
                batch["risk_factor_count"] = pd.to_numeric(
                    batch["risk_factor_count"].map(parse_number), errors="coerce"
                )
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else pd.DataFrame(
                    columns=["id", "cik", "text", "industry", "risk_factor_count"]
                )
            )
            step.set_output(extracted)

        grouped = (
            extracted.groupby("industry")
            .agg(
                avg_risk_factor_count=("risk_factor_count", "mean"),
                filing_count=("risk_factor_count", "count"),
            )
            .reset_index()
        )
        grouped["avg_risk_factor_count"] = pd.to_numeric(
            grouped["avg_risk_factor_count"], errors="coerce"
        ).round(1)
        tracker.record(
            "GROUP_BY(industry, ROUND(AVG(risk_factor_count), 1), COUNT(risk_factor_count))",
            len(extracted),
            len(grouped),
            output=grouped,
        )

        qualifying = grouped[grouped["filing_count"] >= 10].copy()
        tracker.record(
            "FILTER(filing_count >= 10)",
            len(grouped),
            len(qualifying),
            output=qualifying,
        )

        answer_frame = qualifying.sort_values(
            ["avg_risk_factor_count", "industry"],
            ascending=[False, True],
            kind="mergesort",
        ).head(5).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(avg_risk_factor_count DESC, industry ASC) -> LIMIT(5)",
            len(qualifying),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
