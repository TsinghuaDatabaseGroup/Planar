#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-071."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-071"
DIRECTIONS = ("mutual", "unilateral", "unclear")


def _compare_durations(row):
    confidentiality = row["confidentiality_term_years"]
    noncompete = row["noncompete_duration_years"]
    if confidentiality == noncompete:
        return "equal"
    if confidentiality > noncompete:
        return "confidentiality_term_longer"
    return "noncompete_duration_longer"


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
            "SEM_FILTER(document is both NDA and employment/restrictive agreement)",
            input_rows=len(documents),
        ) as step:
            dual_status = documents.sem_filter(
                "The document {text} is both a non-disclosure agreement and an "
                "employment or restrictive-covenant agreement, corresponding to "
                "dual NDA|EX-10 status."
            ).reset_index(drop=True)
            step.set_output(dual_status)

        with tracker.step(
            "SEM_EXTRACT(direction, finite terms, and governing law)",
            input_rows=len(dual_status),
        ) as step:
            extracted = dual_status.sem_extract(
                input_cols=["text"],
                output_cols={
                    "nda_direction": (
                        "exactly mutual, unilateral, or unclear according to the NDA "
                        "nondisclosure duty"
                    ),
                    "confidentiality_term_years": (
                        "the finite confidentiality term normalized to years, or null "
                        "when unstated, perpetual, or unquantifiable"
                    ),
                    "noncompete_duration_years": (
                        "the finite non-compete duration normalized to years, or null "
                        "when unstated or unquantifiable"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                },
            )
            extracted["nda_direction"] = extracted["nda_direction"].map(
                lambda value: normalize_enum(value, DIRECTIONS)
            )
            extracted["confidentiality_term_years"] = extracted[
                "confidentiality_term_years"
            ].map(parse_number)
            extracted["noncompete_duration_years"] = extracted[
                "noncompete_duration_years"
            ].map(parse_number)
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        complete = extracted.loc[
            extracted["confidentiality_term_years"].notna()
            & extracted["noncompete_duration_years"].notna()
            & extracted["governing_law"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(confidentiality term, non-compete duration, and law IS NOT NULL)",
            len(extracted),
            len(complete),
            output=complete,
        )

        projected = complete[
            [
                "document_id",
                "nda_direction",
                "confidentiality_term_years",
                "noncompete_duration_years",
                "governing_law",
            ]
        ].copy()
        projected["duration_comparison"] = complete.apply(
            _compare_durations,
            axis=1,
        )
        projected = projected.reset_index(drop=True)
        tracker.record(
            "PROJECT(document fields and duration comparison)",
            len(complete),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
