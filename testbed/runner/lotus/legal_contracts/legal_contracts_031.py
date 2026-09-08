#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-031."""

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

TASK_ID = "legal_contracts-031"


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
            "SEM_FILTER(compensation recovery or clawback policy)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is a compensation recovery or clawback policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_FILTER(misconduct trigger limited to executive officers)",
            input_rows=len(policies),
        ) as step:
            qualifying = policies.sem_filter(
                "The policy {text} authorizes compensation recovery for employee "
                "misconduct, and the persons covered by that misconduct trigger are "
                "limited to current or former executive officers. For this trigger, "
                "it excludes directors, key managers, other employees, and senior "
                "executives who are not executive officers. All conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(company name)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={"company_name": "the company whose policy this is"},
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
