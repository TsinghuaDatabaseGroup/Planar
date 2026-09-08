#!/usr/bin/env python3
"""LOTUS pipeline for finance-011."""

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
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "finance-011"
BATCH_SIZE = 100
GEOGRAPHIC_LEVELS = (
    "domestic_only",
    "limited_international",
    "globally_diversified",
    "unassigned",
)
ASSIGNED_LEVELS = GEOGRAPHIC_LEVELS[:3]


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2023.csv")[["text"]].copy()
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
            "SEM_EXTRACT(geographic scope and three filing indicators)",
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
                        "geographic_level": (
                            "classify the reporting company's operating footprint as "
                            "exactly one of domestic_only when operations are confined "
                            "to the United States, limited_international when it has "
                            "some but geographically limited operations outside the "
                            "United States, globally_diversified when operations span "
                            "a broad set of countries or regions, or unassigned when "
                            "the scope cannot be determined"
                        ),
                        "foreign_currency_risk": (
                            "true when the filing reports foreign-currency or "
                            "exchange-rate risk; false otherwise"
                        ),
                        "has_international_ops": (
                            "true when the filing describes actual company operations "
                            "outside the United States; false otherwise"
                        ),
                        "has_litigation": (
                            "true when the filing reports litigation, lawsuits, legal "
                            "proceedings, or regulatory investigations involving the "
                            "company; false otherwise"
                        ),
                    },
                )
                batch["geographic_level"] = batch["geographic_level"].map(
                    lambda value: normalize_enum(value, GEOGRAPHIC_LEVELS)
                )
                for column in (
                    "foreign_currency_risk",
                    "has_international_ops",
                    "has_litigation",
                ):
                    batch[column] = batch[column].map(parse_bool)
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else filings.iloc[0:0].assign(
                    geographic_level=pd.Series(dtype="object"),
                    foreign_currency_risk=pd.Series(dtype="bool"),
                    has_international_ops=pd.Series(dtype="bool"),
                    has_litigation=pd.Series(dtype="bool"),
                )
            )
            step.set_output(extracted)

        assigned = extracted[
            extracted["geographic_level"].isin(ASSIGNED_LEVELS)
        ].copy()
        tracker.record(
            "FILTER(geographic_level IN assigned geographic groups)",
            len(extracted),
            len(assigned),
            output=assigned,
        )

        answer_frame = (
            assigned.groupby("geographic_level", sort=False)
            .agg(
                filing_count=("geographic_level", "size"),
                foreign_currency_risk_rate_percent=(
                    "foreign_currency_risk",
                    "mean",
                ),
                international_operations_rate_percent=(
                    "has_international_ops",
                    "mean",
                ),
                litigation_rate_percent=("has_litigation", "mean"),
            )
            .reset_index()
        )
        for column in (
            "foreign_currency_risk_rate_percent",
            "international_operations_rate_percent",
            "litigation_rate_percent",
        ):
            answer_frame[column] = (100.0 * answer_frame[column]).round(1)
        tracker.record(
            "GROUP_BY(geographic_level, filing_count, three rounded rates)",
            len(assigned),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} geographic rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
