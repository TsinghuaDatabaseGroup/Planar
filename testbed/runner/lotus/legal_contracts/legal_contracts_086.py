#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-086."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-086"


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
            "SEM_FILTER(NDA with effective date after 2010-01-01)",
            input_rows=len(documents),
        ) as step:
            dated_ndas = documents.sem_filter(
                "The document {text} is classified as a non-disclosure agreement, "
                "including a dual-labelled NDA exhibit, and has an explicitly stated "
                "effective date strictly after January 1, 2010."
            ).reset_index(drop=True)
            step.set_output(dated_ndas)

        with tracker.step(
            "SEM_FILTER(confidential information includes computer-readable material)",
            input_rows=len(dated_ndas),
        ) as step:
            electronic = dated_ndas.sem_filter(
                "The confidential-information definition in {text} explicitly includes "
                "electronic data or records, email, software or code, metadata, or "
                "other computer-readable material."
            ).reset_index(drop=True)
            step.set_output(electronic)

        with tracker.step(
            "SEM_FILTER(required destruction of at least some materials or copies)",
            input_rows=len(electronic),
        ) as step:
            destruction_required = electronic.sem_filter(
                "The agreement {text} requires destruction of at least some "
                "confidential materials or copies. Exclude clauses that only require "
                "return and clauses that merely allow an unrestricted choice between "
                "return and destruction."
            ).reset_index(drop=True)
            step.set_output(destruction_required)

        with tracker.step(
            "SEM_FILTER(confidentiality or non-use survives termination)",
            input_rows=len(destruction_required),
        ) as step:
            surviving = destruction_required.sem_filter(
                "The confidentiality or non-use obligations in {text} expressly "
                "survive termination or expiry, or otherwise continue beyond the "
                "stated agreement term."
            ).reset_index(drop=True)
            step.set_output(surviving)

        with tracker.step(
            "SEM_EXTRACT(canonical governing law)",
            input_rows=len(surviving),
        ) as step:
            extracted = surviving.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction canonicalized "
                        "to its conventional full English name, or null when unstated"
                    )
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        stated_law = extracted.loc[
            extracted["governing_law"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(governing_law IS NOT NULL)",
            len(extracted),
            len(stated_law),
            output=stated_law,
        )

        grouped = (
            stated_law.groupby("governing_law", sort=False)
            .size()
            .rename("matching_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([governing_law], COUNT(*))",
            len(stated_law),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["matching_count", "governing_law"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(matching_count DESC, governing_law ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
