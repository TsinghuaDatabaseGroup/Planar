#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-081."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    normalize_company_name,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-081"


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

        with tracker.step(
            "SEM_FILTER(genuine EX-97 clawback or compensation-recovery policy)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is a genuine EX-97 clawback or compensation-"
                "recovery policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_FILTER(beyond baseline SEC requirements with misconduct trigger)",
            input_rows=len(policies),
        ) as step:
            enhanced = policies.sem_filter(
                "The policy {text} goes beyond baseline SEC clawback requirements and "
                "includes misconduct as a recovery trigger. Both conditions must hold."
            ).reset_index(drop=True)
            step.set_output(enhanced)

        with tracker.step(
            "SEM_FILTER(coverage beyond executive officers)",
            input_rows=len(enhanced),
        ) as step:
            broader_coverage = enhanced.sem_filter(
                "The policy {text} explicitly extends recovery beyond executive "
                "officers to at least one of non-executive employees, directors, or "
                "key managers."
            ).reset_index(drop=True)
            step.set_output(broader_coverage)

        with tracker.step(
            "SEM_EXTRACT(primary company)",
            input_rows=len(broader_coverage),
        ) as step:
            extracted = broader_coverage.sem_extract(
                input_cols=["text"],
                output_cols={"company": "the primary company name stated in the policy"},
            )
            extracted["company"] = extracted["company"].map(clean_text)
            step.set_output(extracted)

        deduplicated = extracted[["company"]].copy()
        deduplicated["company_key"] = deduplicated["company"].map(
            normalize_company_name
        )
        deduplicated = deduplicated.drop_duplicates(
            "company_key",
            keep="first",
        )[["company"]].reset_index(drop=True)
        tracker.record(
            "DEDUP(NORMALIZE_COMPANY_NAME(company))",
            len(extracted),
            len(deduplicated),
            output=deduplicated,
        )
        answer = df_records(deduplicated)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
