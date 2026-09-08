#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-046."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-046"
DIRECTIONS = ("mutual", "unilateral")


def _normalize_term_years(value):
    normalized = str(value).strip().lower()
    if "perpetual" in normalized or "indefinite" in normalized:
        return "perpetual"
    return parse_number(value)


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
            "SEM_FILTER(Delaware-governed NDA or confidentiality agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a non-disclosure or confidentiality agreement "
                "expressly governed by Delaware law."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(term at least three years or perpetual and six carve-outs)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} has a confidentiality term of at least three "
                "years or a perpetual or indefinite term, and includes all six "
                "operative confidentiality exceptions for publicly available "
                "information, prior knowledge, independent development, lawful "
                "unrestricted third-party receipt, legally compelled disclosure, "
                "and disclosure authorized by the disclosing party."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(term in years and nondisclosure direction)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "term_duration_years": (
                        "the confidentiality term as a number of years for a fixed "
                        "duration, or exactly perpetual for a perpetual or indefinite "
                        "term"
                    ),
                    "non_disclosure_direction": (
                        "exactly mutual if both sides owe the nondisclosure duty, "
                        "or unilateral if only one side owes it"
                    ),
                },
            )
            extracted["term_duration_years"] = extracted[
                "term_duration_years"
            ].map(_normalize_term_years)
            extracted["non_disclosure_direction"] = extracted[
                "non_disclosure_direction"
            ].map(lambda value: normalize_enum(value, DIRECTIONS))
            extracted = extracted[
                ["document_id", "term_duration_years", "non_disclosure_direction"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
