#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-005."""

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

TASK_ID = "legal_contracts-005"


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
            "SEM_FILTER(mutual NDA, unilateral NDA, or confidentiality-and-standstill agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(confidentiality duration and governing law)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "stated_duration": (
                        "the original wording that states the confidentiality "
                        "duration, or null if no duration is stated"
                    ),
                    "duration_months": (
                        "the fixed confidentiality duration normalized to months as "
                        "a number; null when unstated, perpetual, indefinite, or "
                        "otherwise unquantifiable"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                },
            )
            extracted["stated_duration"] = extracted["stated_duration"].map(
                clean_optional_text
            )
            extracted["duration_months"] = extracted["duration_months"].map(
                parse_number
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["duration_months"].notna()
            & (extracted["duration_months"] <= 12)
            & extracted["governing_law"].notna(),
            ["document_id", "stated_duration", "duration_months", "governing_law"],
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(duration_months IS NOT NULL AND duration_months <= 12 AND governing_law IS NOT NULL)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
