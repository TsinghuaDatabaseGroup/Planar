#!/usr/bin/env python3
"""LOTUS pipeline for finance-010."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    iter_selected_texts,
    load_table,
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "finance-010"
BATCH_SIZE = 100


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
            "SEM_EXTRACT(ESG/sustainability and AI/ML mentions)",
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
                        "esg_mentioned": (
                            "true when the filing mentions environmental, social, or "
                            "governance topics or sustainability; false otherwise"
                        ),
                        "ai_ml_mentioned": (
                            "true when the filing contains a qualifying artificial-"
                            "intelligence or machine-learning technology reference in "
                            "the context of the company's business, products, or "
                            "strategy; false otherwise"
                        ),
                    },
                )
                batch["esg_mentioned"] = batch["esg_mentioned"].map(parse_bool)
                batch["ai_ml_mentioned"] = batch["ai_ml_mentioned"].map(parse_bool)
                extracted_parts.append(
                    batch.drop(columns=["document_text"], errors="ignore")
                )
            extracted = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else filings.iloc[0:0].assign(
                    esg_mentioned=pd.Series(dtype="bool"),
                    ai_ml_mentioned=pd.Series(dtype="bool"),
                )
            )
            step.set_output(extracted)

        esg = extracted["esg_mentioned"]
        ai_ml = extracted["ai_ml_mentioned"]
        answer = {
            "both_esg_and_ai_ml": int((esg & ai_ml).sum()),
            "esg_only": int((esg & ~ai_ml).sum()),
            "ai_ml_only": int((~esg & ai_ml).sum()),
            "neither": int((~esg & ~ai_ml).sum()),
        }
        tracker.record(
            "GROUP_BY([], four ESG/AI-ML boolean-combination counts)",
            len(extracted),
            1,
            output=answer,
        )

    print(f"Result: {sum(answer.values())} classified filings")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
