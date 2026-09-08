#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-059."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-059"


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
            "SEM_FILTER(genuine SOX Section 906 certification)",
            input_rows=len(documents),
        ) as step:
            certifications = documents.sem_filter(
                "The document {text} is a genuine Sarbanes-Oxley Section 906 "
                "certification."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_EXTRACT(distinct certifying officers)",
            input_rows=len(certifications),
        ) as step:
            extracted = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "certifying_officers": (
                        "a deduplicated list of the full names of certifying officers "
                        "named as signatories"
                    )
                },
            )
            extracted["certifying_officers"] = extracted[
                "certifying_officers"
            ].map(parse_string_list)
            step.set_output(extracted)

        projected = extracted[["certifying_officers"]].copy()
        projected["officer_count"] = projected["certifying_officers"].map(len)
        projected = projected[["officer_count"]].reset_index(drop=True)
        tracker.record(
            "PROJECT(CARDINALITY(certifying_officers) AS officer_count)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("officer_count", sort=False)
            .size()
            .rename("certification_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([officer_count], COUNT(*))",
            len(projected),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values("officer_count", kind="stable").reset_index(
            drop=True
        )
        tracker.record(
            "ORDER_BY(officer_count ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
