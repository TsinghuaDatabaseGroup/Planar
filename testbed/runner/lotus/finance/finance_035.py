#!/usr/bin/env python3
"""LOTUS pipeline for finance-035."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
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

TASK_ID = "finance-035"
BATCH_SIZE = 100
COMPETITIVE_POSITIONS = (
    "emerging",
    "niche_player",
    "challenger",
    "major_player",
    "market_leader",
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
        filings = load_table("SEC", "CSV/2023.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 60000].copy()
        tracker.record(
            "FILTER(word_count > 60000)",
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
            "SEM_FILTER(acquisition-integration risk)",
            input_rows=len(candidates),
        ) as step:
            integration_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                integration_parts.append(
                    documents.sem_filter(
                        "The 2023 10-K filing {document_text} identifies risk arising "
                        "from integrating an acquired business, operations, systems, "
                        "personnel, products, or technology."
                    )
                )
            integration_risk = (
                pd.concat(integration_parts, ignore_index=True)
                if integration_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(integration_risk))

        with tracker.step(
            "SEM_FILTER(litigation or legal proceedings)",
            input_rows=len(integration_risk),
        ) as step:
            litigation = sem_filter_in_batches(
                integration_risk,
                "The 2023 10-K filing {document_text} reports pending, threatened, "
                "or ongoing litigation, legal proceedings, lawsuits, or regulatory "
                "investigations involving the reporting company.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(litigation))

        with tracker.step(
            "SEM_CLASSIFY(competitive-position category)",
            input_rows=len(litigation),
        ) as step:
            classified = sem_map_in_batches(
                litigation,
                "Assign the reporting company's competitive position from the 2023 "
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

        with tracker.step(
            "SEM_EXTRACT(company, industry, workforce, and M&A transaction count)",
            input_rows=len(classified),
        ) as step:
            extracted = sem_extract_in_batches(
                classified,
                input_cols=["document_text"],
                output_cols={
                    "company_name": (
                        "the reporting company's legal name stated in the filing"
                    ),
                    "industry_sector": (
                        "the reporting company's industry sector stated or clearly "
                        "supported by the filing"
                    ),
                    "employee_count": (
                        "an explicitly reported company-wide total employee count as "
                        "an integer, or null when no such total is disclosed"
                    ),
                    "ma_transaction_count": (
                        "the number of distinct merger or acquisition transactions "
                        "mentioned in the filing as an integer"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted["company_name"] = extracted["company_name"].map(clean_text)
            extracted["industry_sector"] = extracted["industry_sector"].map(
                clean_text
            )
            for column in ("employee_count", "ma_transaction_count"):
                extracted[column] = pd.to_numeric(
                    extracted[column].map(parse_number), errors="coerce"
                )
            step.set_output(_intermediate_view(extracted))

        qualifying = extracted[
            extracted["competitive_position"].notna()
            & (extracted["competitive_position"] != "undetermined")
            & (extracted["ma_transaction_count"] >= 3)
            & extracted["employee_count"].notna()
        ].copy()
        tracker.record(
            "FILTER(competitive_position != 'undetermined' AND ma_transaction_count >= 3 AND employee_count IS NOT NULL)",
            len(extracted),
            len(qualifying),
            output=_intermediate_view(qualifying),
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
            .copy()
        )
        tracker.record(
            "DEDUP(company_name, keep=max_by(employee_count, ma_transaction_count, word_count, cik))",
            len(qualifying),
            len(deduped),
            output=_intermediate_view(deduped),
        )

        ranked = deduped.assign(
            _company_sort=deduped["company_name"].str.lower()
        ).sort_values(
            ["employee_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)
        answer_frame = ranked[
            [
                "company_name",
                "industry_sector",
                "competitive_position",
                "employee_count",
                "ma_transaction_count",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(employee_count DESC, LOWER(company_name) ASC) -> LIMIT(10) -> PROJECT",
            len(deduped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
