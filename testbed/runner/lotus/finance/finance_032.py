#!/usr/bin/env python3
"""LOTUS pipeline for finance-032."""

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
    sem_extract_in_batches,
    setup,
)

TASK_ID = "finance-032"
BATCH_SIZE = 100


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _intermediate_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=["document_text", "document_text_2019", "document_text_2023"],
        errors="ignore",
    )


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
            "SEM_FILTER(early-stage or emerging competitor in 2019)",
            input_rows=len(filings_2019),
        ) as step:
            emerging_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings_2019,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                emerging_parts.append(
                    documents.sem_filter(
                        "The 2019 10-K filing {document_text} describes the reporting "
                        "company as an early-stage or emerging competitor in 2019."
                    )
                )
            emerging_2019 = (
                pd.concat(emerging_parts, ignore_index=True)
                if emerging_parts
                else _empty_documents(filings_2019)
            )
            step.set_output(_intermediate_view(emerging_2019))

        with tracker.step(
            "SEM_EXTRACT(reported employee count in 2019)",
            input_rows=len(emerging_2019),
        ) as step:
            employees_2019 = sem_extract_in_batches(
                emerging_2019,
                input_cols=["document_text"],
                output_cols={
                    "employee_count_2019": (
                        "the explicitly reported company-wide total employee count "
                        "as an integer, or null when no explicit total is disclosed"
                    )
                },
                batch_size=BATCH_SIZE,
            )
            employees_2019["employee_count_2019"] = pd.to_numeric(
                employees_2019["employee_count_2019"].map(parse_number),
                errors="coerce",
            )
            step.set_output(_intermediate_view(employees_2019))

        filings_2023 = load_table("SEC", "CSV/2023.csv")[["cik", "text"]].copy()
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv AS filings_2023)",
            None,
            len(filings_2023),
            output=filings_2023,
        )
        tracker.record(
            "SCAN_DOCS(selector=filings_2023.text)",
            len(filings_2023),
            len(filings_2023),
            output=filings_2023,
        )

        with tracker.step(
            "SEM_FILTER(recognized industry or market leader in 2023)",
            input_rows=len(filings_2023),
        ) as step:
            leader_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings_2023,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                leader_parts.append(
                    documents.sem_filter(
                        "The 2023 10-K filing {document_text} describes the reporting "
                        "company as a recognized industry or market leader in 2023."
                    )
                )
            leaders_2023 = (
                pd.concat(leader_parts, ignore_index=True)
                if leader_parts
                else _empty_documents(filings_2023)
            )
            step.set_output(_intermediate_view(leaders_2023))

        with tracker.step(
            "SEM_EXTRACT(2023 company, industry, and employee count)",
            input_rows=len(leaders_2023),
        ) as step:
            employees_2023 = sem_extract_in_batches(
                leaders_2023,
                input_cols=["document_text"],
                output_cols={
                    "company_name": (
                        "the reporting company's legal name stated in the filing"
                    ),
                    "industry_sector": (
                        "the reporting company's industry sector stated or clearly "
                        "supported by the filing"
                    ),
                    "employee_count_2023": (
                        "the explicitly reported company-wide total employee count "
                        "as an integer"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            employees_2023["company_name"] = employees_2023["company_name"].map(
                clean_text
            )
            employees_2023["industry_sector"] = employees_2023[
                "industry_sector"
            ].map(clean_text)
            employees_2023["employee_count_2023"] = pd.to_numeric(
                employees_2023["employee_count_2023"].map(parse_number),
                errors="coerce",
            )
            step.set_output(_intermediate_view(employees_2023))

        matched = employees_2019.merge(
            employees_2023,
            on="cik",
            how="inner",
            sort=False,
            suffixes=("_2019", "_2023"),
        )
        tracker.record(
            "JOIN(inner, emerging_2019.cik = leaders_2023.cik)",
            {"left": len(employees_2019), "right": len(employees_2023)},
            len(matched),
            output=_intermediate_view(matched),
        )

        answer_frame = matched[
            [
                "company_name",
                "industry_sector",
                "employee_count_2019",
                "employee_count_2023",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(company_name, industry_sector, employee_count_2019, employee_count_2023)",
            len(matched),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
