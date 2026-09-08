#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-057."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-057"


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
            "SEM_FILTER(employment or standalone restrictive-covenant agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is an employment agreement or a standalone "
                "restrictive-covenant agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(non-compete >= 4 years with geographic scope)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} contains an operative non-compete restriction "
                "lasting at least four years and expressly states its geographic "
                "scope. Both conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(duration, geography, and prohibited activity)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "duration_years": (
                        "the operative non-compete duration normalized to years as a "
                        "number"
                    ),
                    "geographic_scope": (
                        "the expressly stated geographic scope of the non-compete"
                    ),
                    "prohibited_activity_scope": (
                        "a concise description of the activities prohibited by the "
                        "non-compete"
                    ),
                },
            )
            extracted["duration_years"] = extracted["duration_years"].map(
                parse_number
            )
            extracted["geographic_scope"] = extracted["geographic_scope"].map(
                clean_text
            )
            extracted["prohibited_activity_scope"] = extracted[
                "prohibited_activity_scope"
            ].map(clean_text)
            extracted = extracted[
                [
                    "document_id",
                    "duration_years",
                    "geographic_scope",
                    "prohibited_activity_scope",
                ]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
