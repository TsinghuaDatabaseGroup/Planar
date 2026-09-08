#!/usr/bin/env python3
"""Palimpzest pipeline for finance-032."""

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

TASK_ID = "finance-032"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_text", "document_text_2019", "document_text_2023"],
        errors="ignore",
    )


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
        )[["cik", "document_text"]]
        tracker.record(
            "scan",
            len(filings_2019),
            _intermediate_view(documents_2019),
        )

        emerging_plan = memory_dataset(
            f"{TASK_ID}-emerging-2019", documents_2019
        ).sem_filter(
            filter=(
                "Keep the filing only if it describes the reporting company as an "
                "early-stage or emerging competitor in 2019."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        emerging_result = emerging_plan.run(config)
        emerging_2019 = result_frame(emerging_result, documents_2019)
        tracker.record_semantic(
            "sem_filter",
            len(documents_2019),
            _intermediate_view(emerging_2019),
            emerging_result,
            time.time() - started,
        )

        extraction_2019_plan = memory_dataset(
            f"{TASK_ID}-employees-2019", emerging_2019
        ).sem_map(
            cols=[
                {
                    "name": "employee_count_2019",
                    "type": int,
                    "desc": (
                        "The explicitly reported company-wide total employee count, "
                        "or null when no explicit total is disclosed."
                    ),
                }
            ],
            desc="Extract the reported employee count.",
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_2019_result = extraction_2019_plan.run(config)
        employees_2019 = result_frame(
            extraction_2019_result,
            emerging_2019,
            ["employee_count_2019"],
        )
        employees_2019["employee_count_2019"] = pd.to_numeric(
            employees_2019["employee_count_2019"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(emerging_2019),
            _intermediate_view(employees_2019),
            extraction_2019_result,
            time.time() - started,
        )

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
        )[["cik", "document_text"]]
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
        leaders_2023 = result_frame(leader_result, documents_2023)
        tracker.record_semantic(
            "sem_filter",
            len(documents_2023),
            _intermediate_view(leaders_2023),
            leader_result,
            time.time() - started,
        )

        extraction_2023_plan = memory_dataset(
            f"{TASK_ID}-leader-details-2023", leaders_2023
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
                    "desc": "The reporting company's industry sector.",
                },
                {
                    "name": "employee_count_2023",
                    "type": int,
                    "desc": (
                        "The explicitly reported company-wide total employee count."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name, industry sector, and reported "
                "employee count."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_2023_result = extraction_2023_plan.run(config)
        employees_2023 = result_frame(
            extraction_2023_result,
            leaders_2023,
            ["company_name", "industry_sector", "employee_count_2023"],
        )
        employees_2023["employee_count_2023"] = pd.to_numeric(
            employees_2023["employee_count_2023"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(leaders_2023),
            _intermediate_view(employees_2023),
            extraction_2023_result,
            time.time() - started,
        )

        matched = employees_2019.loc[employees_2019["cik"].notna()].merge(
            employees_2023.loc[employees_2023["cik"].notna()],
            on="cik",
            how="inner",
            sort=False,
            suffixes=("_2019", "_2023"),
        )
        tracker.record(
            "join",
            {"left": len(employees_2019), "right": len(employees_2023)},
            _intermediate_view(matched),
        )

        answer_frame = matched[
            [
                "company_name",
                "industry_sector",
                "employee_count_2019",
                "employee_count_2023",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(matched), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
