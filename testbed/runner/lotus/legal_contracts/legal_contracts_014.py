#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-014."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-014"
CLAUSE_COLUMNS = (
    "has_internal_controls_clause",
    "has_fraud_disclosure_clause",
    "has_material_changes_clause",
)


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
            "SEM_FILTER(genuine SOX Section 302 certification)",
            input_rows=len(documents),
        ) as step:
            certifications = documents.sem_filter(
                "The document {text} is a genuine Sarbanes-Oxley Section 302 "
                "certification."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_EXTRACT(Section 302 clause indicators)",
            input_rows=len(certifications),
        ) as step:
            extracted = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "has_internal_controls_clause": (
                        "true if the certification contains the internal-controls "
                        "responsibility clause; false otherwise"
                    ),
                    "has_fraud_disclosure_clause": (
                        "true if the certification contains the management-fraud "
                        "disclosure clause; false otherwise"
                    ),
                    "has_material_changes_clause": (
                        "true if the certification contains the material-changes "
                        "clause; false otherwise"
                    ),
                },
            )
            for column in CLAUSE_COLUMNS:
                extracted[column] = extracted[column].map(parse_bool)
            step.set_output(extracted)

        grouped = (
            extracted.groupby(list(CLAUSE_COLUMNS), sort=False, dropna=False)
            .size()
            .rename("certification_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(three clause indicators, COUNT(*))",
            len(extracted),
            len(grouped),
            output=grouped,
        )

        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
