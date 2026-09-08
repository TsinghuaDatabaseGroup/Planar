#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-044."""

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

TASK_ID = "legal_contracts-044"


def main():
    setup(max_tokens=768, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(EX-97 clawback or recoupment policy)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is an EX-97 compensation clawback or recoupment "
                "policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_FILTER(intentional misconduct, beyond SEC minimum, and effective date)",
            input_rows=len(policies),
        ) as step:
            qualifying = policies.sem_filter(
                "The policy {text} treats intentional misconduct as a recovery "
                "trigger, extends recovery beyond SEC minimum requirements, and "
                "explicitly states an effective date. All conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(name, company, date, and covered persons)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "document_name": "the stated name or title of the policy",
                    "company_name": "the company whose policy this is",
                    "effective_date": (
                        "the explicitly stated effective date formatted as YYYY-MM-DD"
                    ),
                    "covered_persons": (
                        "a concise description of the persons covered by the policy"
                    ),
                },
            )
            for column in (
                "document_name",
                "company_name",
                "effective_date",
                "covered_persons",
            ):
                extracted[column] = extracted[column].map(clean_text)
            extracted = extracted[
                ["document_name", "company_name", "effective_date", "covered_persons"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
