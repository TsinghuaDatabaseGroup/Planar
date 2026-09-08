#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-045."""

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

TASK_ID = "legal_contracts-045"


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
            "SEM_FILTER(Delaware- or New York-governed NDA)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a non-disclosure agreement expressly governed "
                "by Delaware or New York law."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_FILTER(standstill, six carve-outs, and stated terms)",
            input_rows=len(agreements),
        ) as step:
            qualifying = agreements.sem_filter(
                "The agreement {text} includes an operative standstill provision, "
                "recognizes all six standard confidentiality exceptions for publicly "
                "available information, prior knowledge, independent development, "
                "lawful unrestricted third-party receipt, legally compelled "
                "disclosure, and disclosure authorized by the disclosing party, "
                "states a confidentiality term, and states a quantifiable standstill "
                "period. All conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(name, state, confidentiality term, and standstill period)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "document_name": "the stated name or title of the agreement",
                    "governing_state": (
                        "the qualifying governing state written as its full name"
                    ),
                    "term_duration_years": (
                        "the confidentiality term as a number of years for a fixed "
                        "duration, or exactly perpetual for a perpetual or indefinite "
                        "term"
                    ),
                    "standstill_period_years": (
                        "the quantifiable standstill period normalized to years as a "
                        "number"
                    ),
                },
            )
            extracted["document_name"] = extracted["document_name"].map(clean_text)
            extracted["governing_state"] = extracted["governing_state"].map(clean_text)
            extracted["term_duration_years"] = extracted[
                "term_duration_years"
            ].map(_normalize_term_years)
            extracted["standstill_period_years"] = extracted[
                "standstill_period_years"
            ].map(parse_number)
            extracted = extracted[
                [
                    "document_name",
                    "governing_state",
                    "term_duration_years",
                    "standstill_period_years",
                ]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
