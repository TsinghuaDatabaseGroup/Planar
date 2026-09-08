#!/usr/bin/env python3
"""Palimpzest pipeline for finance-035."""

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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "finance-035"
DATASET = "SEC"
COMPETITIVE_POSITIONS = (
    "emerging",
    "niche_player",
    "challenger",
    "major_player",
    "market_leader",
    "undetermined",
)


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2023.csv",
            ["cik", "text", "word_count"],
        )
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[filings["word_count"] > 60_000].reset_index(
            drop=True
        )
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["cik", "word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        integration_plan = memory_dataset(
            f"{TASK_ID}-integration-risk", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it identifies risk arising from "
                "integrating an acquired business, operations, systems, personnel, "
                "products, or technology."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        integration_result = integration_plan.run(config)
        integration_risk = result_frame(integration_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(integration_risk),
            integration_result,
            time.time() - started,
        )

        litigation_plan = memory_dataset(
            f"{TASK_ID}-litigation", integration_risk
        ).sem_filter(
            filter=(
                "Keep the filing only if it reports pending, threatened, or "
                "ongoing litigation, legal proceedings, lawsuits, or regulatory "
                "investigations involving the reporting company."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        litigation_result = litigation_plan.run(config)
        litigation = result_frame(litigation_result, integration_risk)
        tracker.record_semantic(
            "sem_filter",
            len(integration_risk),
            _intermediate_view(litigation),
            litigation_result,
            time.time() - started,
        )

        position_plan = memory_dataset(
            f"{TASK_ID}-competitive-position", litigation
        ).sem_map(
            cols=[
                {
                    "name": "competitive_position",
                    "type": str,
                    "desc": (
                        "Exactly one category: emerging, niche_player, challenger, "
                        "major_player, market_leader, or undetermined when it cannot "
                        "be determined."
                    ),
                }
            ],
            desc="Assign the filing to its competitive-position category.",
            depends_on=["document_text"],
        )
        started = time.time()
        position_result = position_plan.run(config)
        classified = result_frame(
            position_result,
            litigation,
            ["competitive_position"],
        )
        classified["competitive_position"] = classified[
            "competitive_position"
        ].map(lambda value: normalize_enum(value, COMPETITIVE_POSITIONS))
        tracker.record_semantic(
            "sem_map",
            len(litigation),
            _intermediate_view(classified),
            position_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-company-details", classified
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
                    "name": "employee_count",
                    "type": int,
                    "desc": (
                        "An explicitly reported company-wide total employee count, "
                        "or null when no such total is disclosed."
                    ),
                },
                {
                    "name": "ma_transaction_count",
                    "type": int,
                    "desc": (
                        "The number of distinct merger or acquisition transactions "
                        "mentioned in the filing."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name, industry sector, reported employee "
                "count, and number of distinct merger or acquisition transactions."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            classified,
            [
                "company_name",
                "industry_sector",
                "employee_count",
                "ma_transaction_count",
            ],
        )
        for column in ("employee_count", "ma_transaction_count"):
            extracted[column] = pd.to_numeric(
                extracted[column], errors="coerce"
            ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(classified),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["competitive_position"].notna()
            & extracted["competitive_position"].ne("undetermined")
            & extracted["ma_transaction_count"].ge(3)
            & extracted["employee_count"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(extracted),
            _intermediate_view(qualifying),
        )

        deduped = (
            qualifying.sort_values(
                [
                    "company_name",
                    "employee_count",
                    "ma_transaction_count",
                    "word_count",
                    "cik",
                ],
                ascending=[True, False, False, False, False],
                kind="mergesort",
            )
            .drop_duplicates("company_name", keep="first")
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(qualifying), _intermediate_view(deduped))

        ordered = deduped.assign(
            _company_sort=deduped["company_name"].astype(str).str.casefold()
        ).sort_values(
            ["employee_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        )
        tracker.record("order_by", len(deduped), _intermediate_view(ordered))

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), _intermediate_view(limited))

        answer_frame = limited[
            [
                "company_name",
                "industry_sector",
                "competitive_position",
                "employee_count",
                "ma_transaction_count",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
