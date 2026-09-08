#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-083."""

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

TASK_ID = "legal_contracts-083"


def _sorted_distinct_rounded(values):
    return sorted({round(float(value), 4) for value in values})


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
            "SEM_FILTER(standalone NDA excluding NDA|EX-10)",
            input_rows=len(documents),
        ) as step:
            standalone_ndas = documents.sem_filter(
                "The document {text} is a standalone non-disclosure agreement and is "
                "not also an EX-10 employment or restrictive-covenant agreement."
            ).reset_index(drop=True)
            step.set_output(standalone_ndas)

        with tracker.step(
            "SEM_FILTER(all six confidentiality exceptions)",
            input_rows=len(standalone_ndas),
        ) as step:
            complete_exceptions = standalone_ndas.sem_filter(
                "The agreement {text} includes all six operative confidentiality "
                "exceptions for public information, prior knowledge, independent "
                "development, lawful unrestricted third-party receipt, legal "
                "compulsion, and disclosure with consent."
            ).reset_index(drop=True)
            step.set_output(complete_exceptions)

        with tracker.step(
            "SEM_EXTRACT(finite confidentiality term and governing law)",
            input_rows=len(complete_exceptions),
        ) as step:
            extracted = complete_exceptions.sem_extract(
                input_cols=["text"],
                output_cols={
                    "term_years": (
                        "the finite confidentiality term normalized to years, or null "
                        "when absent, perpetual, or unquantifiable"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                },
            )
            extracted["term_years"] = extracted["term_years"].map(parse_number)
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        complete = extracted.loc[
            extracted["term_years"].notna()
            & extracted["governing_law"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(term_years IS NOT NULL AND governing_law IS NOT NULL)",
            len(extracted),
            len(complete),
            output=complete,
        )

        grouped = (
            complete.groupby("governing_law", sort=False)
            .agg(
                agreement_count=("document_id", "size"),
                term_lengths_years=("term_years", _sorted_distinct_rounded),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([governing_law], count, sorted distinct term lengths)",
            len(complete),
            len(grouped),
            output=grouped,
        )

        frequent = grouped.loc[grouped["agreement_count"] >= 3].reset_index(drop=True)
        tracker.record(
            "FILTER(agreement_count >= 3)",
            len(grouped),
            len(frequent),
            output=frequent,
        )

        ordered = frequent.sort_values(
            ["agreement_count", "governing_law"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(agreement_count DESC, governing_law ASC)",
            len(frequent),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
