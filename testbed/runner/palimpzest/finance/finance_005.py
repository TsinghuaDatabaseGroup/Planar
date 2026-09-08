#!/usr/bin/env python3
"""Palimpzest pipeline for finance-005."""

from __future__ import annotations

import os
import re
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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "finance-005"
DATASET = "SEC"
BLANK_CHECK_ALIASES = {
    "blank_check",
    "blank_check_company",
    "blank_check_companies",
    "spac",
    "spacs",
    "special_purpose_acquisition_company",
    "special_purpose_acquisition_companies",
}


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def _canonical_sector(value) -> str:
    label = re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().casefold()
    ).strip("_")
    return "blank_check_companies" if label in BLANK_CHECK_ALIASES else label


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(DATASET, "CSV/2022.csv", ["text"])
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-filing-profile", documents
        ).sem_map(
            cols=[
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": (
                        "A concise canonical lowercase snake_case industry-sector "
                        "label. Normalize case, separators, and synonymous labels, "
                        "including blank-check, SPAC, and special-purpose acquisition "
                        "company variants as blank_check_companies."
                    ),
                },
                {
                    "name": "has_going_concern_doubt",
                    "type": bool,
                    "desc": (
                        "True only when the filing expresses substantial doubt about "
                        "the reporting company's ability to continue as a going "
                        "concern; false otherwise."
                    ),
                },
            ],
            desc=(
                "Extract a canonical industry sector and whether the filing "
                "expresses going-concern doubt."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            ["industry_sector", "has_going_concern_doubt"],
        )
        extracted["industry_sector"] = extracted["industry_sector"].map(
            _canonical_sector
        )
        extracted["has_going_concern_doubt"] = extracted[
            "has_going_concern_doubt"
        ].map(parse_bool)
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby("industry_sector", sort=False)
            .agg(
                filing_count=("industry_sector", "size"),
                going_concern_count=("has_going_concern_doubt", "sum"),
            )
            .reset_index()
        )
        grouped["unrounded_rate"] = (
            grouped["going_concern_count"] / grouped["filing_count"]
        )
        grouped["going_concern_rate_percent"] = (
            100.0 * grouped["unrounded_rate"]
        ).round(1)
        tracker.record("groupby", len(extracted), grouped)

        qualifying = grouped.loc[grouped["filing_count"] >= 20].reset_index(
            drop=True
        )
        tracker.record("filter", len(grouped), qualifying)

        ordered = qualifying.sort_values(
            ["unrounded_rate", "industry_sector"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record("order_by", len(qualifying), ordered)

        limited = ordered.head(3).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited[
            ["industry_sector", "going_concern_rate_percent", "filing_count"]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
