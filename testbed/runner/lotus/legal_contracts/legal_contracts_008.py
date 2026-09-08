#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-008."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    clean_text,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-008"


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
            "SEM_FILTER(operative non-compete restriction)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The agreement {text} contains an operative non-compete restriction, "
                "rather than an incidental reference to non-competition."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(name, non-compete period, geographic scope, and law)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "document_name": "the stated name or title of the agreement",
                    "non_compete_months": (
                        "the operative non-compete duration normalized to months, or "
                        "null when unstated or unquantifiable"
                    ),
                    "geographic_scope": (
                        "the geographic scope stated for the non-compete restriction, "
                        "or null when unstated"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                },
            )
            extracted["document_name"] = extracted["document_name"].map(clean_text)
            extracted["non_compete_months"] = extracted[
                "non_compete_months"
            ].map(parse_number)
            extracted["geographic_scope"] = extracted["geographic_scope"].map(
                clean_optional_text
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["non_compete_months"].notna()
            & (extracted["non_compete_months"] <= 6)
            & extracted["geographic_scope"].notna()
            & extracted["governing_law"].notna(),
            [
                "document_name",
                "non_compete_months",
                "geographic_scope",
                "governing_law",
            ],
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(non_compete_months IS NOT NULL AND non_compete_months <= 6 AND geographic_scope IS NOT NULL AND governing_law IS NOT NULL)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
