#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-003."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-003"
PERIOD_TYPES = ("annual", "quarterly", "other")


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
            "SEM_FILTER(genuine Section 302 certification ending June 30, 2025)",
            input_rows=len(documents),
        ) as step:
            certifications = documents.sem_filter(
                "The document {text} is a genuine Sarbanes-Oxley Section 302 "
                "certification and explicitly states that the reporting period "
                "ended on June 30, 2025. A signature date, filing date, or other "
                "date does not satisfy the reporting-period requirement."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_EXTRACT(certifying officers and reporting-period type)",
            input_rows=len(certifications),
        ) as step:
            extracted = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "certifying_officers": (
                        "a list of the full names of the officers who certify the "
                        "document"
                    ),
                    "reporting_period_type": (
                        "exactly annual, quarterly, or other according to the "
                        "reporting period covered by the certification"
                    ),
                },
            )
            extracted["certifying_officers"] = extracted[
                "certifying_officers"
            ].map(parse_string_list)
            extracted["reporting_period_type"] = extracted[
                "reporting_period_type"
            ].map(lambda value: normalize_enum(value, PERIOD_TYPES))
            extracted = extracted[
                ["document_id", "certifying_officers", "reporting_period_type"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
