#!/usr/bin/env python3
"""LOTUS pipeline for finance-015."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_selected_texts,
    load_table,
    save_output,
    setup,
)

TASK_ID = "finance-015"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2024.csv")[
            ["id", "cik", "text", "word_count"]
        ].copy()
        filings["word_count"] = filings["word_count"].astype(int)
        tracker.record(
            "SCAN_TABLE(CSV/2024.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 50000].copy()
        tracker.record(
            "FILTER(word_count > 50000)",
            len(filings),
            len(candidates),
            output=candidates,
        )

        documents = load_selected_texts(
            "SEC",
            candidates,
            path_column="text",
            output_column="document_text",
        )
        tracker.record(
            "SCAN_DOCS(selector=filtered.text)",
            len(candidates),
            len(documents),
            output=candidates,
        )

        with tracker.step(
            "SEM_FILTER(operational use of AI or machine learning)",
            input_rows=len(documents),
        ) as step:
            if documents.empty:
                ai_filings = documents.copy()
            else:
                ai_filings = documents.sem_filter(
                    "The 10-K filing {document_text} describes operational use of "
                    "artificial intelligence or machine learning by the company."
                )
            step.set_output(ai_filings.drop(columns=["document_text"], errors="ignore"))

        with tracker.step(
            "SEM_FILTER(no substantial doubt about continuing as a going concern)",
            input_rows=len(ai_filings),
        ) as step:
            if ai_filings.empty:
                continuing = ai_filings.copy()
            else:
                continuing = ai_filings.sem_filter(
                    "The 10-K filing {document_text} does not report substantial "
                    "doubt about the company's ability to continue as a going concern."
                )
            step.set_output(continuing.drop(columns=["document_text"], errors="ignore"))

        with tracker.step(
            "SEM_EXTRACT(company legal name and industry sector)",
            input_rows=len(continuing),
        ) as step:
            if continuing.empty:
                extracted = continuing.copy()
                extracted["company_name"] = None
                extracted["industry"] = None
            else:
                extracted = continuing.sem_extract(
                    input_cols=["document_text"],
                    output_cols={
                        "company_name": "the company's legal name as reported in the filing",
                        "industry": "the company's primary industry sector",
                    },
                )
                extracted["company_name"] = extracted["company_name"].map(clean_text)
                extracted["industry"] = extracted["industry"].map(clean_text)
            step.set_output(extracted.drop(columns=["document_text"], errors="ignore"))

        ranked = extracted.assign(
            _company_sort=extracted["company_name"].astype(str).str.lower()
        ).sort_values(
            ["word_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)
        answer_frame = ranked[["company_name", "industry", "word_count"]].reset_index(
            drop=True
        )
        tracker.record(
            "ORDER_BY(word_count DESC, company_name ASC) -> LIMIT(10)",
            len(extracted),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
