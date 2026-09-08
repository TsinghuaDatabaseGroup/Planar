#!/usr/bin/env python3
"""LOTUS pipeline for finance-005."""

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
    normalize_free_label,
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "finance-005"
BATCH_SIZE = 100
BLANK_CHECK_ALIASES = {
    "blank_check",
    "blank_check_company",
    "blank_check_companies",
    "spac",
    "spacs",
    "special_purpose_acquisition_company",
    "special_purpose_acquisition_companies",
}


def _canonical_sector(value) -> str:
    label = normalize_free_label(value)
    return "blank_check_companies" if label in BLANK_CHECK_ALIASES else label


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2022.csv")[["text"]].copy()
        tracker.record(
            "SCAN_TABLE(CSV/2022.csv)", None, len(filings), output=filings
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_EXTRACT(canonical industry sector and going-concern doubt)",
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
                        "industry_sector": (
                            "a concise canonical industry-sector label for the "
                            "reporting company; use lowercase snake_case and normalize "
                            "synonymous labels, including blank-check, SPAC, and "
                            "special-purpose acquisition company variants as "
                            "blank_check_companies"
                        ),
                        "has_going_concern_doubt": (
                            "true only when the filing expresses substantial doubt "
                            "about the reporting company's ability to continue as a "
                            "going concern; false otherwise"
                        ),
                    },
                )
                batch["industry_sector"] = batch["industry_sector"].map(
                    _canonical_sector
                )
                batch["has_going_concern_doubt"] = batch[
                    "has_going_concern_doubt"
                ].map(parse_bool)
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else filings.iloc[0:0].assign(
                    industry_sector=pd.Series(dtype="object"),
                    has_going_concern_doubt=pd.Series(dtype="bool"),
                )
            )
            step.set_output(extracted)

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
        tracker.record(
            "GROUP_BY(industry_sector, filing_count, going_concern_count, unrounded_rate, rounded_percent)",
            len(extracted),
            len(grouped),
            output=grouped,
        )

        qualifying = grouped[grouped["filing_count"] >= 20].copy()
        tracker.record(
            "FILTER(filing_count >= 20)",
            len(grouped),
            len(qualifying),
            output=qualifying,
        )

        ranked = qualifying.sort_values(
            ["unrounded_rate", "industry_sector"],
            ascending=[False, True],
            kind="mergesort",
        ).head(3)
        answer_frame = ranked[
            ["industry_sector", "going_concern_rate_percent", "filing_count"]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(unrounded_rate DESC, industry_sector ASC) -> LIMIT(3) -> PROJECT",
            len(qualifying),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
