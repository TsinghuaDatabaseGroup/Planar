#!/usr/bin/env python3
"""LOTUS pipeline for finance-001."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_selected_texts,
    load_table,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "finance-001"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2024.csv")[["id", "cik", "text"]].copy()
        tracker.record(
            "SCAN_TABLE(CSV/2024.csv)", None, len(filings), output=filings
        )

        documents = load_selected_texts(
            "SEC",
            filings,
            path_column="text",
            output_column="document_text",
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(documents),
            output=filings,
        )

        with tracker.step(
            "SEM_FILTER(company operates in a retail industry)",
            input_rows=len(documents),
        ) as step:
            if documents.empty:
                retail = documents.copy()
            else:
                retail = documents.sem_filter(
                    "The 10-K filing {document_text} describes a company operating "
                    "in a retail industry."
                )
            step.set_output(retail.drop(columns=["document_text"], errors="ignore"))

        with tracker.step(
            "SEM_EXTRACT(company name, employee count, retail subsector)",
            input_rows=len(retail),
        ) as step:
            if retail.empty:
                extracted = retail.copy()
                extracted["company_name"] = None
                extracted["employee_count"] = pd.Series(dtype="Int64")
                extracted["retail_subsector"] = None
            else:
                extracted = retail.sem_extract(
                    input_cols=["document_text"],
                    output_cols={
                        "company_name": "the company's legal name as reported in the filing",
                        "employee_count": "the total reported employee count as an integer",
                        "retail_subsector": "the company's specific retail subsector",
                    },
                )
                extracted["company_name"] = extracted["company_name"].map(clean_text)
                extracted["employee_count"] = extracted["employee_count"].map(
                    parse_number
                )
                extracted["employee_count"] = pd.to_numeric(
                    extracted["employee_count"], errors="coerce"
                ).astype("Int64")
                extracted["retail_subsector"] = extracted["retail_subsector"].map(
                    clean_text
                )
            step.set_output(extracted.drop(columns=["document_text"], errors="ignore"))

        ranked = extracted.assign(
            _company_sort=extracted["company_name"].astype(str).str.lower()
        ).sort_values(
            ["employee_count", "_company_sort"],
            ascending=[False, True],
            na_position="last",
            kind="mergesort",
        ).head(3)
        answer_frame = ranked[
            ["company_name", "employee_count", "retail_subsector"]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(employee_count DESC, company_name ASC) -> LIMIT(3)",
            len(extracted),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
