#!/usr/bin/env python3
"""LOTUS pipeline for finance-020."""

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

TASK_ID = "finance-020"
BATCH_SIZE = 100
INDUSTRY_LABELS = {
    "pharmaceutical": "Pharmaceutical",
    "software": "Software",
    "other": "Other",
}


def _normalize_industry(value) -> str | None:
    normalized = normalize_enum(value, INDUSTRY_LABELS)
    return INDUSTRY_LABELS.get(normalized) if normalized is not None else None


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
            "SEM_CLASSIFY(industry sector for Pharmaceutical/Software comparison)",
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
                    "Assign the reporting company in the 2023 10-K filing "
                    "{document_text} to exactly one industry label for this "
                    "comparison. Output exactly one label: Pharmaceutical, Software, "
                    "or Other.",
                    suffix="industry_sector",
                )
                batch["industry_sector"] = batch["industry_sector"].map(
                    _normalize_industry
                )
                classified_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            classified = (
                pd.concat(classified_parts, ignore_index=True)
                if classified_parts
                else filings.iloc[0:0].assign(
                    industry_sector=pd.Series(dtype="object")
                )
            )
            step.set_output(classified)

        selected = classified[
            classified["industry_sector"].isin(["Pharmaceutical", "Software"])
        ].copy()
        tracker.record(
            "FILTER(industry_sector IN ['Pharmaceutical', 'Software'])",
            len(classified),
            len(selected),
            output=selected,
        )

        with tracker.step(
            "SEM_EXTRACT(AI/ML, going concern, international operations, and litigation)",
            input_rows=len(selected),
        ) as step:
            extracted_parts = []
            for documents in iter_selected_texts(
                "SEC",
                selected,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["document_text"],
                    output_cols={
                        "ai_ml_mentioned": (
                            "true when the filing contains a qualifying artificial-"
                            "intelligence or machine-learning technology reference in "
                            "the context of the company's business, products, or "
                            "strategy; false otherwise"
                        ),
                        "has_going_concern": (
                            "true when the filing expresses substantial doubt about "
                            "the company's ability to continue as a going concern; "
                            "false otherwise"
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
                for column in (
                    "ai_ml_mentioned",
                    "has_going_concern",
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
                else selected.iloc[0:0].assign(
                    ai_ml_mentioned=pd.Series(dtype="bool"),
                    has_going_concern=pd.Series(dtype="bool"),
                    has_international_ops=pd.Series(dtype="bool"),
                    has_litigation=pd.Series(dtype="bool"),
                )
            )
            step.set_output(extracted)

        answer_frame = (
            extracted.groupby("industry_sector", sort=False)
            .agg(
                filing_count=("industry_sector", "size"),
                ai_ml_mention_rate_percent=("ai_ml_mentioned", "mean"),
                going_concern_rate_percent=("has_going_concern", "mean"),
                international_operations_rate_percent=(
                    "has_international_ops",
                    "mean",
                ),
                litigation_rate_percent=("has_litigation", "mean"),
            )
            .reset_index()
        )
        for column in (
            "ai_ml_mention_rate_percent",
            "going_concern_rate_percent",
            "international_operations_rate_percent",
            "litigation_rate_percent",
        ):
            answer_frame[column] = (100.0 * answer_frame[column]).round(1)
        tracker.record(
            "GROUP_BY(industry_sector, filing_count, four rounded rates)",
            len(extracted),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} industry rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
