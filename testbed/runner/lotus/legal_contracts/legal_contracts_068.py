#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-068."""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-068"
EXCEPTION_COLUMNS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "lawful_third_party_receipt",
    "legal_compulsion",
    "consent",
)


def _normalize_title(value):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def main():
    setup(max_tokens=1024, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(in-scope agreement with identifiable title)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement with an identifiable agreement title."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(title, exceptions, return requirement, and relief)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "agreement_title": "the identifiable stated agreement title",
                    "public_information": (
                        "true if an operative public-information exception is present"
                    ),
                    "prior_knowledge": (
                        "true if an operative prior-knowledge exception is present"
                    ),
                    "independent_development": (
                        "true if an operative independent-development exception is present"
                    ),
                    "lawful_third_party_receipt": (
                        "true if an operative lawful unrestricted third-party-receipt "
                        "exception is present"
                    ),
                    "legal_compulsion": (
                        "true if an operative legal-compulsion exception is present"
                    ),
                    "consent": (
                        "true if an operative disclosure-with-consent exception is present"
                    ),
                    "return_of_materials": (
                        "true if return or destruction of confidential materials is "
                        "required"
                    ),
                    "injunctive_relief": (
                        "true if injunctive or equitable relief is provided"
                    ),
                },
            )
            extracted["agreement_title"] = extracted["agreement_title"].map(clean_text)
            for column in (*EXCEPTION_COLUMNS, "return_of_materials", "injunctive_relief"):
                extracted[column] = extracted[column].map(parse_bool)
            step.set_output(extracted)

        projected = extracted[["agreement_title"]].copy()
        projected["normalized_agreement_title"] = extracted["agreement_title"].map(
            _normalize_title
        )
        projected["exception_count"] = extracted[list(EXCEPTION_COLUMNS)].sum(axis=1)
        projected["has_return_and_injunctive"] = (
            extracted["return_of_materials"] & extracted["injunctive_relief"]
        ).astype(int)
        projected["has_all_six"] = (
            projected["exception_count"] == 6
        ).astype(int)
        projected = projected[
            [
                "normalized_agreement_title",
                "exception_count",
                "has_return_and_injunctive",
                "has_all_six",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(normalized title and protection-profile measures)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("normalized_agreement_title", sort=False)
            .agg(
                family_size=("normalized_agreement_title", "size"),
                avg_exception_count=("exception_count", "mean"),
                return_and_injunctive_share=("has_return_and_injunctive", "mean"),
                all_six_exceptions_share=("has_all_six", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_exception_count"] = grouped[
                "avg_exception_count"
            ].round(2)
            grouped["return_and_injunctive_share"] = grouped[
                "return_and_injunctive_share"
            ].round(4)
            grouped["all_six_exceptions_share"] = grouped[
                "all_six_exceptions_share"
            ].round(4)
        tracker.record(
            "GROUP_BY([normalized title], family profile statistics)",
            len(projected),
            len(grouped),
            output=grouped,
        )

        large_families = grouped.loc[grouped["family_size"] >= 5].reset_index(
            drop=True
        )
        tracker.record(
            "FILTER(family_size >= 5)",
            len(grouped),
            len(large_families),
            output=large_families,
        )

        ordered = large_families.sort_values(
            ["family_size", "normalized_agreement_title"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(family_size DESC, normalized title ASC)",
            len(large_families),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
