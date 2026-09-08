#!/usr/bin/env python3
"""Palimpzest pipeline for finance-038."""

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

TASK_ID = "finance-038"
DATASET = "SEC"
COMPETITIVE_POSITIONS = (
    "emerging",
    "niche_player",
    "challenger",
    "major_player",
    "market_leader",
    "undetermined",
)
RISK_THEMES = (
    "debt_financing",
    "regulatory_compliance",
    "litigation_legal",
    "key_personnel",
    "customer_concentration",
    "intellectual_property",
    "cybersecurity",
)


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2022.csv",
            ["text", "word_count"],
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[filings["word_count"] > 65_000].reset_index(
            drop=True
        )
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        concern_plan = memory_dataset(
            f"{TASK_ID}-going-concern", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it expresses substantial doubt about the "
                "reporting company's ability to continue as a going concern."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        concern_result = concern_plan.run(config)
        concern_filings = result_frame(concern_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(concern_filings),
            concern_result,
            time.time() - started,
        )

        international_plan = memory_dataset(
            f"{TASK_ID}-international", concern_filings
        ).sem_filter(
            filter=(
                "Keep the filing only if it describes actual company operations "
                "outside the United States."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        international_result = international_plan.run(config)
        international = result_frame(international_result, concern_filings)
        tracker.record_semantic(
            "sem_filter",
            len(concern_filings),
            _intermediate_view(international),
            international_result,
            time.time() - started,
        )

        position_plan = memory_dataset(
            f"{TASK_ID}-competitive-position", international
        ).sem_map(
            cols=[
                {
                    "name": "competitive_position",
                    "type": str,
                    "desc": (
                        "Exactly one category: emerging, niche_player, challenger, "
                        "major_player, market_leader, or undetermined."
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
            international,
            ["competitive_position"],
        )
        classified["competitive_position"] = classified[
            "competitive_position"
        ].map(lambda value: normalize_enum(value, COMPETITIVE_POSITIONS))
        tracker.record_semantic(
            "sem_map",
            len(international),
            _intermediate_view(classified),
            position_result,
            time.time() - started,
        )

        determined = classified.loc[
            classified["competitive_position"].notna()
            & classified["competitive_position"].ne("undetermined")
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), _intermediate_view(determined))

        employee_plan = memory_dataset(
            f"{TASK_ID}-employees", determined
        ).sem_map(
            cols=[
                {
                    "name": "employee_count",
                    "type": int,
                    "desc": (
                        "An explicitly reported company-wide total employee count, "
                        "or null when no such total is disclosed."
                    ),
                }
            ],
            desc="Extract the reported employee count.",
            depends_on=["document_text"],
        )
        started = time.time()
        employee_result = employee_plan.run(config)
        qualified_filings = result_frame(
            employee_result,
            determined,
            ["employee_count"],
        )
        qualified_filings["employee_count"] = pd.to_numeric(
            qualified_filings["employee_count"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(determined),
            _intermediate_view(qualified_filings),
            employee_result,
            time.time() - started,
        )

        position_stats = (
            qualified_filings.groupby("competitive_position", sort=False)
            .agg(
                filing_count=("competitive_position", "size"),
                median_employee_count=("employee_count", "median"),
            )
            .reset_index()
        )
        position_stats["median_employee_count"] = pd.to_numeric(
            position_stats["median_employee_count"], errors="coerce"
        ).round(1)
        tracker.record("groupby", len(qualified_filings), position_stats)

        theme_rows = []
        for index, (position, group) in enumerate(
            qualified_filings.groupby("competitive_position", sort=False),
            start=1,
        ):
            aggregate_input = group[["document_text"]].reset_index(drop=True)
            aggregate_plan = memory_dataset(
                f"{TASK_ID}-risk-theme-{index}", aggregate_input
            ).sem_agg(
                col={
                    "name": "top_risk_theme",
                    "type": str,
                    "desc": "The most prevalent risk-theme label for this group.",
                },
                agg=(
                    "Across these filings, identify the most prevalent risk theme. "
                    "Output exactly one label: debt_financing, "
                    "regulatory_compliance, litigation_legal, key_personnel, "
                    "customer_concentration, intellectual_property, or "
                    "cybersecurity. Break frequency ties by choosing the "
                    "alphabetically earliest label."
                ),
                depends_on=["document_text"],
            )
            started = time.time()
            aggregate_result = aggregate_plan.run(config)
            aggregate_frame = result_frame(aggregate_result)
            tracker.record_semantic(
                "sem_agg",
                len(aggregate_input),
                aggregate_frame,
                aggregate_result,
                time.time() - started,
            )
            if aggregate_frame.empty:
                raise ValueError(
                    f"{TASK_ID}: semantic aggregate returned no risk theme"
                )
            theme_rows.append(
                {
                    "competitive_position": position,
                    "top_risk_theme": normalize_enum(
                        aggregate_frame.iloc[0]["top_risk_theme"],
                        RISK_THEMES,
                    ),
                }
            )
        theme_summary = pd.DataFrame(
            theme_rows,
            columns=["competitive_position", "top_risk_theme"],
        )

        combined = position_stats.merge(
            theme_summary,
            on="competitive_position",
            how="inner",
            sort=False,
        )
        tracker.record(
            "join",
            {"left": len(position_stats), "right": len(theme_summary)},
            combined,
        )

        qualifying = combined.loc[combined["filing_count"] >= 10].reset_index(
            drop=True
        )
        tracker.record("filter", len(combined), qualifying)

        projected = qualifying[
            [
                "competitive_position",
                "filing_count",
                "median_employee_count",
                "top_risk_theme",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(qualifying), projected)

        ordered = projected.sort_values(
            ["filing_count", "competitive_position"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(projected), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
