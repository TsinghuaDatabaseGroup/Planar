#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-042."""

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
    parse_optional_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-042"


def _normalize_term_years(value):
    if "perpetual" in str(value).strip().lower() or "indefinite" in str(value).strip().lower():
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
            "SEM_FILTER(mutual non-disclosure agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(term, law, and non-compete or employee non-solicitation)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} expressly states a confidentiality term and "
                "a governing law, and contains at least one operative non-compete or "
                "employee non-solicitation clause."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(term in years, law, and clause indicators)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "term_duration_years": (
                        "the confidentiality term normalized to a number of years "
                        "for a fixed duration, or exactly perpetual for an indefinite "
                        "or perpetual term"
                    ),
                    "governing_law": "the expressly stated governing-law jurisdiction",
                    "non_compete_present": (
                        "true if an operative non-compete clause is present; false "
                        "otherwise"
                    ),
                    "non_solicitation_employees_present": (
                        "true if an operative employee non-solicitation clause is "
                        "present; false otherwise"
                    ),
                },
            )
            extracted["term_duration_years"] = extracted[
                "term_duration_years"
            ].map(_normalize_term_years)
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted["non_compete_present"] = extracted[
                "non_compete_present"
            ].map(parse_optional_bool)
            extracted["non_solicitation_employees_present"] = extracted[
                "non_solicitation_employees_present"
            ].map(parse_optional_bool)
            extracted = extracted[
                [
                    "document_id",
                    "term_duration_years",
                    "governing_law",
                    "non_compete_present",
                    "non_solicitation_employees_present",
                ]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
