#!/usr/bin/env python3
"""LOTUS pipeline for finance-038."""

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

TASK_ID = "finance-038"
BATCH_SIZE = 100
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
        filings = load_table("SEC", "CSV/2022.csv")[["text", "word_count"]].copy()
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2022.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 65000].copy()
        tracker.record(
            "FILTER(word_count > 65000)",
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
            "SEM_FILTER(substantial going-concern doubt)",
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
                        "The 2022 10-K filing {document_text} expresses substantial "
                        "doubt about the reporting company's ability to continue as "
                        "a going concern."
                    )
                )
            concern_filings = (
                pd.concat(concern_parts, ignore_index=True)
                if concern_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(concern_filings))

        with tracker.step(
            "SEM_FILTER(operations outside the United States)",
            input_rows=len(concern_filings),
        ) as step:
            international = sem_filter_in_batches(
                concern_filings,
                "The 2022 10-K filing {document_text} describes actual company "
                "operations outside the United States.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(international))

        with tracker.step(
            "SEM_CLASSIFY(competitive-position category)",
            input_rows=len(international),
        ) as step:
            classified = sem_map_in_batches(
                international,
                "Assign the reporting company's competitive position from the 2022 "
                "10-K filing {document_text}. Output exactly one label: emerging, "
                "niche_player, challenger, major_player, market_leader, or "
                "undetermined when it cannot be determined.",
                suffix="competitive_position",
                batch_size=BATCH_SIZE,
            )
            classified["competitive_position"] = classified[
                "competitive_position"
            ].map(lambda value: normalize_enum(value, COMPETITIVE_POSITIONS))
            step.set_output(_intermediate_view(classified))

        determined = classified[
            classified["competitive_position"].notna()
            & (classified["competitive_position"] != "undetermined")
        ].copy()
        tracker.record(
            "FILTER(competitive_position != 'undetermined')",
            len(classified),
            len(determined),
            output=_intermediate_view(determined),
        )

        with tracker.step(
            "SEM_EXTRACT(reported employee count)",
            input_rows=len(determined),
        ) as step:
            qualified_filings = sem_extract_in_batches(
                determined,
                input_cols=["document_text"],
                output_cols={
                    "employee_count": (
                        "an explicitly reported company-wide total employee count as "
                        "an integer, or null when no such total is disclosed"
                    )
                },
                batch_size=BATCH_SIZE,
            )
            qualified_filings["employee_count"] = pd.to_numeric(
                qualified_filings["employee_count"].map(parse_number),
                errors="coerce",
            )
            step.set_output(_intermediate_view(qualified_filings))

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
        tracker.record(
            "GROUP_BY(competitive_position, filing_count, rounded median employee count)",
            len(qualified_filings),
            len(position_stats),
            output=position_stats,
        )

        with tracker.step(
            "SEM_AGGREGATE(top risk theme by competitive position)",
            input_rows=len(qualified_filings),
        ) as step:
            if qualified_filings.empty:
                theme_summary = pd.DataFrame(
                    columns=["competitive_position", "top_risk_theme"]
                )
            else:
                theme_summary = qualified_filings[
                    ["competitive_position", "document_text"]
                ].sem_agg(
                    "Across the filings in this competitive-position group, identify "
                    "the most prevalent risk theme represented in {document_text}. "
                    "Output exactly one label: debt_financing, "
                    "regulatory_compliance, litigation_legal, key_personnel, "
                    "customer_concentration, intellectual_property, or cybersecurity. "
                    "Break frequency ties by choosing the alphabetically earliest "
                    "label.",
                    suffix="top_risk_theme",
                    group_by=["competitive_position"],
                )
                theme_summary["top_risk_theme"] = theme_summary[
                    "top_risk_theme"
                ].map(lambda value: normalize_enum(value, RISK_THEMES))
            step.set_output(theme_summary)

        combined = position_stats.merge(
            theme_summary,
            on="competitive_position",
            how="inner",
            sort=False,
        )
        tracker.record(
            "JOIN(inner, position_stats.competitive_position = theme_summary.competitive_position)",
            {"left": len(position_stats), "right": len(theme_summary)},
            len(combined),
            output=combined,
        )

        qualifying = combined[combined["filing_count"] >= 10].copy()
        tracker.record(
            "FILTER(filing_count >= 10)",
            len(combined),
            len(qualifying),
            output=qualifying,
        )

        answer_frame = qualifying[
            [
                "competitive_position",
                "filing_count",
                "median_employee_count",
                "top_risk_theme",
            ]
        ].sort_values(
            ["filing_count", "competitive_position"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tracker.record(
            "PROJECT(position statistics and top theme) -> ORDER_BY(filing_count DESC, competitive_position ASC)",
            len(qualifying),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
