#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-007."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-007"


def main():
    setup(max_tokens=512, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all)",
            None,
            len(documents),
            output=documents,
        )

        long_documents = documents.loc[
            documents["word_count"] > 7500
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(word_count > 7500)",
            len(documents),
            len(long_documents),
            output=long_documents,
        )

        with tracker.step(
            "SEM_FILTER(insider-trading policy extending restrictions to family)",
            input_rows=len(long_documents),
        ) as step:
            policies = long_documents.sem_filter(
                "The document {text} is a substantive insider-trading policy and "
                "explicitly extends its trading restrictions to family or household "
                "members of covered persons."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(company name)",
            input_rows=len(policies),
        ) as step:
            extracted = policies.sem_extract(
                input_cols=["text"],
                output_cols={"company_name": "the company name stated in the policy"},
            )
            extracted["company_name"] = extracted["company_name"].map(clean_text)
            extracted = extracted[
                ["document_id", "company_name"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
