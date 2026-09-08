#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-029."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-029"


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
            "SEM_FILTER(operative non-compete restriction)",
            input_rows=len(documents),
        ) as step:
            non_compete_agreements = documents.sem_filter(
                "The agreement {text} contains an operative non-compete restriction."
            ).reset_index(drop=True)
            step.set_output(non_compete_agreements)

        with tracker.step(
            "SEM_FILTER(no operative standstill restriction)",
            input_rows=len(non_compete_agreements),
        ) as step:
            no_standstill = non_compete_agreements.sem_filter(
                "The agreement {text} contains no operative standstill restriction."
            ).reset_index(drop=True)
            step.set_output(no_standstill)

        with tracker.step(
            "SEM_EXTRACT(governing law and non-compete duration)",
            input_rows=len(no_standstill),
        ) as step:
            extracted = no_standstill.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                    "non_compete_months": (
                        "the operative non-compete duration normalized to months as "
                        "a number, or null when unstated or unquantifiable"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            extracted["non_compete_months"] = extracted[
                "non_compete_months"
            ].map(parse_number)
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["governing_law"].notna()
            & extracted["non_compete_months"].notna()
            & (extracted["non_compete_months"] <= 6),
            ["document_id", "governing_law", "non_compete_months"],
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(governing_law IS NOT NULL AND non_compete_months IS NOT NULL AND non_compete_months <= 6)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
