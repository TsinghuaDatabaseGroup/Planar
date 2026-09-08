#!/usr/bin/env python3
"""Palimpzest pipeline for finance-020."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "finance-020"
DATASET = "SEC"
INDUSTRY_LABELS = {
    "pharmaceutical": "Pharmaceutical",
    "software": "Software",
    "other": "Other",
}


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def _normalize_industry(value) -> str | None:
    normalized = normalize_enum(value, INDUSTRY_LABELS)
    return INDUSTRY_LABELS.get(normalized) if normalized is not None else None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(DATASET, "CSV/2023.csv", ["text"])
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        classification_plan = memory_dataset(
            f"{TASK_ID}-industry", documents
        ).sem_map(
            cols=[
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": "Exactly one label: Pharmaceutical, Software, or Other.",
                }
            ],
            desc="Assign the filing to its industry sector for this comparison.",
            depends_on=["document_text"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            documents,
            ["industry_sector"],
        )
        classified["industry_sector"] = classified["industry_sector"].map(
            _normalize_industry
        )
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(classified),
            classification_result,
            time.time() - started,
        )

        selected = classified.loc[
            classified["industry_sector"].isin(["Pharmaceutical", "Software"])
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), _intermediate_view(selected))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-indicators", selected
        ).sem_map(
            cols=[
                {
                    "name": "ai_ml_mentioned",
                    "type": bool,
                    "desc": (
                        "True when the filing contains a qualifying AI/ML technology "
                        "reference in the context of the company's business, "
                        "products, or strategy; false otherwise."
                    ),
                },
                {
                    "name": "has_going_concern",
                    "type": bool,
                    "desc": (
                        "True when the filing expresses substantial doubt about the "
                        "company's ability to continue as a going concern; false "
                        "otherwise."
                    ),
                },
                {
                    "name": "has_international_ops",
                    "type": bool,
                    "desc": (
                        "True when the filing describes actual company operations "
                        "outside the United States; false otherwise."
                    ),
                },
                {
                    "name": "has_litigation",
                    "type": bool,
                    "desc": (
                        "True when the filing reports litigation, lawsuits, legal "
                        "proceedings, or regulatory investigations involving the "
                        "company; false otherwise."
                    ),
                },
            ],
            desc=(
                "Determine whether the filing contains a qualifying AI/ML "
                "reference, going-concern doubt, international operations, and "
                "litigation or legal proceedings."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            selected,
            [
                "ai_ml_mentioned",
                "has_going_concern",
                "has_international_ops",
                "has_litigation",
            ],
        )
        for column in (
            "ai_ml_mentioned",
            "has_going_concern",
            "has_international_ops",
            "has_litigation",
        ):
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map",
            len(selected),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        grouped = (
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
            grouped[column] = (100.0 * grouped[column]).round(1)
        tracker.record("groupby", len(extracted), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
