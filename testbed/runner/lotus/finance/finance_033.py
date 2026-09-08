#!/usr/bin/env python3
"""LOTUS pipeline for finance-033."""

import os
import sys
from decimal import ROUND_HALF_UP, Decimal

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

TASK_ID = "finance-033"
BATCH_SIZE = 100


def _employee_change_percent(previous, current) -> float | None:
    if pd.isna(previous) or pd.isna(current) or int(previous) == 0:
        return None
    change = (
        Decimal(100)
        * (Decimal(int(current)) - Decimal(int(previous)))
        / Decimal(int(previous))
    )
    return float(change.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings_2019 = load_table("SEC", "CSV/2019.csv")[["cik", "text"]].copy()
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2019.csv AS filings_2019)",
            None,
            len(filings_2019),
            output=filings_2019,
        )
        tracker.record(
            "SCAN_DOCS(selector=filings_2019.text)",
            len(filings_2019),
            len(filings_2019),
            output=filings_2019,
        )

        with tracker.step(
            "SEM_EXTRACT(explicit company-wide employee count in 2019)",
            input_rows=len(filings_2019),
        ) as step:
            extracted_2019_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings_2019,
                path_column="text",
                output_column="document_2019",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["document_2019"],
                    output_cols={
                        "employee_count_2019": (
                            "only an explicitly disclosed company-wide total employee "
                            "count as an integer; return null for partial workforce "
                            "figures, constructed totals, or non-headcount numbers"
                        )
                    },
                )
                batch["employee_count_2019"] = pd.to_numeric(
                    batch["employee_count_2019"].map(parse_number),
                    errors="coerce",
                )
                extracted_2019_parts.append(
                    batch.drop(columns=["document_2019"], errors="ignore")
                )
            employees_2019 = (
                pd.concat(extracted_2019_parts, ignore_index=True)
                if extracted_2019_parts
                else filings_2019.iloc[0:0].assign(
                    employee_count_2019=pd.Series(dtype="float64")
                )
            )
            step.set_output(employees_2019)

        eligible_2019 = employees_2019[
            employees_2019["employee_count_2019"] >= 1000
        ].copy()
        tracker.record(
            "FILTER(employee_count_2019 >= 1000)",
            len(employees_2019),
            len(eligible_2019),
            output=eligible_2019,
        )

        filings_2023 = load_table("SEC", "CSV/2023.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        filings_2023["word_count"] = pd.to_numeric(
            filings_2023["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv AS filings_2023)",
            None,
            len(filings_2023),
            output=filings_2023,
        )

        primary_2023 = (
            filings_2023.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .rename(columns={"text": "primary_text_2023"})
        )
        tracker.record(
            "GROUP_BY(cik, MAX_BY(text, word_count) AS primary_text_2023)",
            len(filings_2023),
            len(primary_2023),
            output=primary_2023,
        )
        tracker.record(
            "SCAN_DOCS(selector=primary_text_2023)",
            len(primary_2023),
            len(primary_2023),
            output=primary_2023,
        )

        with tracker.step(
            "SEM_EXTRACT(2023 company and explicit company-wide employee count)",
            input_rows=len(primary_2023),
        ) as step:
            extracted_2023_parts = []
            for documents in iter_selected_texts(
                "SEC",
                primary_2023,
                path_column="primary_text_2023",
                output_column="document_2023",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["document_2023"],
                    output_cols={
                        "company_name": (
                            "the reporting company's legal name stated in the filing"
                        ),
                        "employee_count_2023": (
                            "only an explicitly disclosed company-wide total employee "
                            "count as an integer; return null for partial workforce "
                            "figures, constructed totals, or non-headcount numbers"
                        ),
                    },
                )
                batch["company_name"] = batch["company_name"].map(clean_text)
                batch["employee_count_2023"] = pd.to_numeric(
                    batch["employee_count_2023"].map(parse_number),
                    errors="coerce",
                )
                extracted_2023_parts.append(
                    batch.drop(columns=["document_2023"], errors="ignore")
                )
            employees_2023 = (
                pd.concat(extracted_2023_parts, ignore_index=True)
                if extracted_2023_parts
                else primary_2023.iloc[0:0].assign(
                    company_name=pd.Series(dtype="object"),
                    employee_count_2023=pd.Series(dtype="float64"),
                )
            )
            step.set_output(employees_2023)

        paired = eligible_2019.merge(
            employees_2023,
            on="cik",
            how="inner",
            sort=False,
            suffixes=("_2019", "_2023"),
        )
        tracker.record(
            "JOIN(inner, eligible_2019.cik = employees_2023.cik)",
            {"left": len(eligible_2019), "right": len(employees_2023)},
            len(paired),
            output=paired,
        )

        changes = paired[
            ["company_name", "employee_count_2019", "employee_count_2023"]
        ].copy()
        changes["employee_change_percent"] = [
            _employee_change_percent(previous, current)
            for previous, current in zip(
                changes["employee_count_2019"],
                changes["employee_count_2023"],
            )
        ]
        tracker.record(
            "PROJECT(employee counts and rounded percentage change)",
            len(paired),
            len(changes),
            output=changes,
        )

        positive = changes[
            changes["employee_change_percent"].notna()
            & (changes["employee_change_percent"] > 0)
        ].copy()
        tracker.record(
            "FILTER(employee_change_percent > 0)",
            len(changes),
            len(positive),
            output=positive,
        )
        top_growth = positive.sort_values(
            ["employee_change_percent", "company_name"],
            ascending=[False, True],
            kind="mergesort",
        ).head(5).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(employee_change_percent DESC, company_name ASC) -> LIMIT(5)",
            len(positive),
            len(top_growth),
            output=top_growth,
        )

        negative = changes[
            changes["employee_change_percent"].notna()
            & (changes["employee_change_percent"] < 0)
        ].copy()
        tracker.record(
            "FILTER(employee_change_percent < 0)",
            len(changes),
            len(negative),
            output=negative,
        )
        top_decline = negative.sort_values(
            ["employee_change_percent", "company_name"],
            ascending=[True, True],
            kind="mergesort",
        ).head(5).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(employee_change_percent ASC, company_name ASC) -> LIMIT(5)",
            len(negative),
            len(top_decline),
            output=top_decline,
        )

        answer = {
            "top_5_growth": df_records(top_growth),
            "top_5_decline": df_records(top_decline),
        }
        tracker.record(
            "PROJECT(top_5_growth, top_5_decline)",
            {"growth": len(top_growth), "decline": len(top_decline)},
            1,
            output=answer,
        )

    print(
        f"Result: {len(top_growth)} growth rows, "
        f"{len(top_decline)} decline rows"
    )
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
