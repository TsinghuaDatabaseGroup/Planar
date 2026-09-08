#!/usr/bin/env python3
"""LOTUS pipeline for finance-026."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    iter_selected_texts,
    load_table,
    normalize_enum,
    parse_number,
    save_output,
    sem_extract_in_batches,
    sem_filter_in_batches,
    sem_map_in_batches,
    setup,
)

TASK_ID = "finance-026"
BATCH_SIZE = 100
GROWTH_STRATEGIES = (
    "organic",
    "acquisition_driven",
    "hybrid",
    "not_applicable",
    "undetermined",
)


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _intermediate_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["document_text"], errors="ignore")


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2020.csv")[
            ["id", "cik", "text", "word_count"]
        ].copy()
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2020.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 50000].copy()
        tracker.record(
            "FILTER(word_count > 50000)",
            len(filings),
            len(candidates),
            output=candidates,
        )
        tracker.record(
            "SCAN_DOCS(selector=filtered.text)",
            len(candidates),
            len(candidates),
            output=candidates,
        )

        with tracker.step(
            "SEM_FILTER(substantial doubt about continuing as a going concern)",
            input_rows=len(candidates),
        ) as step:
            concern_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                concern_parts.append(
                    documents.sem_filter(
                        "The 10-K filing {document_text} expresses substantial doubt "
                        "about the company's ability to continue as a going concern."
                    )
                )
            concern_filings = (
                pd.concat(concern_parts, ignore_index=True)
                if concern_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(concern_filings))

        with tracker.step(
            "SEM_FILTER(litigation or legal proceedings)",
            input_rows=len(concern_filings),
        ) as step:
            litigation = sem_filter_in_batches(
                concern_filings,
                "The 10-K filing {document_text} reports pending, threatened, or "
                "ongoing litigation, legal proceedings, lawsuits, or regulatory "
                "investigations.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(litigation))

        with tracker.step(
            "SEM_CLASSIFY(stated growth-strategy category)",
            input_rows=len(litigation),
        ) as step:
            classified = sem_map_in_batches(
                litigation,
                "Classify the stated growth strategy in the 10-K filing "
                "{document_text}. Output exactly one label: organic when growth is "
                "primarily through internal business expansion; acquisition_driven "
                "when it is primarily through acquisitions; hybrid when both are "
                "stated components; not_applicable when the filing indicates that no "
                "growth strategy applies; or undetermined when it cannot be determined.",
                suffix="growth_strategy",
                batch_size=BATCH_SIZE,
            )
            classified["growth_strategy"] = classified["growth_strategy"].map(
                lambda value: normalize_enum(value, GROWTH_STRATEGIES)
            )
            step.set_output(_intermediate_view(classified))

        determined = classified[
            classified["growth_strategy"].notna()
            & (classified["growth_strategy"] != "undetermined")
        ].copy()
        tracker.record(
            "FILTER(growth_strategy != 'undetermined')",
            len(classified),
            len(determined),
            output=_intermediate_view(determined),
        )

        with tracker.step(
            "SEM_EXTRACT(employee count and total risk-factor count)",
            input_rows=len(determined),
        ) as step:
            extracted = sem_extract_in_batches(
                determined,
                input_cols=["document_text"],
                output_cols={
                    "employee_count": (
                        "the total reported employee count as an integer, or null when "
                        "it cannot be determined"
                    ),
                    "risk_factor_count": (
                        "the total number of risk factors disclosed in the filing's "
                        "Risk Factors section as an integer, or null when it cannot "
                        "be determined reliably"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            for column in ("employee_count", "risk_factor_count"):
                extracted[column] = pd.to_numeric(
                    extracted[column].map(parse_number), errors="coerce"
                )
            step.set_output(_intermediate_view(extracted))

        grouped = (
            extracted.groupby("growth_strategy", dropna=False)
            .agg(
                filing_count=("id", "size"),
                median_employee_count=("employee_count", "median"),
                median_risk_factor_count=("risk_factor_count", "median"),
            )
            .reset_index()
        )
        grouped["median_employee_count"] = grouped["median_employee_count"].round(1)
        grouped["median_risk_factor_count"] = grouped[
            "median_risk_factor_count"
        ].round(1)
        tracker.record(
            "GROUP_BY(growth_strategy, COUNT, MEDIAN(employee_count), MEDIAN(risk_factor_count))",
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
            ["filing_count", "growth_strategy"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(filing_count DESC, growth_strategy ASC)",
            len(qualifying),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
