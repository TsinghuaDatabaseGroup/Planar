#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-105."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_iso_date,
    parse_bool,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-105"
CORE_COLUMNS = [
    "has_confidential_information_definition",
    "has_nondisclosure",
    "has_non_use",
    "has_return_of_materials",
    "has_survival",
]
EXCEPTION_COLUMNS = [
    "has_public_information_exception",
    "has_prior_knowledge_exception",
    "has_independent_development_exception",
    "has_third_party_receipt_exception",
    "has_legal_compulsion_exception",
    "has_consent_exception",
]


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(document has NDA status, including dual-status exhibits)",
            input_rows=len(documents),
        ) as step:
            ndas = documents.sem_filter(
                "The document {text} has non-disclosure-agreement status, including a "
                "dual-status SEC exhibit."
            ).reset_index(drop=True)
            step.set_output(ndas)

        with tracker.step(
            "SEM_EXTRACT(effective date and governing-law jurisdictions)",
            input_rows=len(ndas),
        ) as step:
            dated = ndas.sem_extract(
                input_cols=["text"],
                output_cols={
                    "effective_date": (
                        "the explicit agreement effective date in YYYY-MM-DD format, or "
                        "null if none is stated"
                    ),
                    "governing_law_jurisdictions": (
                        "a JSON list containing every expressly stated governing-law "
                        "jurisdiction"
                    ),
                },
            )
            dated["effective_date"] = dated["effective_date"].map(normalize_iso_date)
            dated["governing_law_jurisdictions"] = dated[
                "governing_law_jurisdictions"
            ].map(parse_string_list)
            step.set_output(dated)

        parsed_dates = pd.to_datetime(dated["effective_date"], errors="coerce")
        eligible = dated[
            (parsed_dates >= pd.Timestamp("2010-01-01"))
            & (dated["governing_law_jurisdictions"].map(len) == 1)
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(effective date >= 2010-01-01 and one jurisdiction)",
            len(dated),
            len(eligible),
            output=eligible,
        )

        with tracker.step(
            "SEM_EXTRACT(core clauses, standard exceptions, and perpetual term)",
            input_rows=len(eligible),
        ) as step:
            features = eligible.sem_extract(
                input_cols=["text"],
                output_cols={
                    "has_confidential_information_definition": (
                        "true only if confidential information is explicitly defined"
                    ),
                    "has_nondisclosure": (
                        "true only if an operative nondisclosure obligation is present"
                    ),
                    "has_non_use": (
                        "true only if an operative non-use obligation is present"
                    ),
                    "has_return_of_materials": (
                        "true only if an operative return-of-materials obligation is "
                        "present"
                    ),
                    "has_survival": (
                        "true only if an operative survival clause is present"
                    ),
                    "has_public_information_exception": (
                        "true only if a public-information exception is present"
                    ),
                    "has_prior_knowledge_exception": (
                        "true only if a prior-knowledge exception is present"
                    ),
                    "has_independent_development_exception": (
                        "true only if an independent-development exception is present"
                    ),
                    "has_third_party_receipt_exception": (
                        "true only if a third-party-receipt exception is present"
                    ),
                    "has_legal_compulsion_exception": (
                        "true only if a legal-compulsion exception is present"
                    ),
                    "has_consent_exception": (
                        "true only if a consent exception is present"
                    ),
                    "perpetual_confidentiality_term": (
                        "true only if the confidentiality term is perpetual"
                    ),
                },
            )
            for column in [
                *CORE_COLUMNS,
                *EXCEPTION_COLUMNS,
                "perpetual_confidentiality_term",
            ]:
                features[column] = features[column].map(parse_bool)
            step.set_output(features)

        projected = pd.DataFrame(
            {
                "jurisdiction": features["governing_law_jurisdictions"].map(
                    lambda values: values[0]
                ),
                "all_five_core_clauses": features[CORE_COLUMNS].all(axis=1),
                "all_six_standard_exceptions": features[EXCEPTION_COLUMNS].all(axis=1),
                "perpetual_confidentiality_term": features[
                    "perpetual_confidentiality_term"
                ],
            }
        )
        tracker.record(
            "PROJECT(jurisdiction and three derived indicators)",
            len(features),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("jurisdiction", sort=False)
            .agg(
                nda_count=("jurisdiction", "size"),
                all_five_core_clauses_count=("all_five_core_clauses", "sum"),
                all_six_standard_exceptions_count=(
                    "all_six_standard_exceptions",
                    "sum",
                ),
                perpetual_term_count=("perpetual_confidentiality_term", "sum"),
            )
            .reset_index()
        )
        for column in (
            "nda_count",
            "all_five_core_clauses_count",
            "all_six_standard_exceptions_count",
            "perpetual_term_count",
        ):
            grouped[column] = grouped[column].astype(int)
        tracker.record(
            "GROUP_BY(jurisdiction and four counts)",
            len(projected),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            "nda_count",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(nda_count DESC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(3).reset_index(drop=True)
        tracker.record(
            "LIMIT(3)",
            len(ordered),
            len(limited),
            output=limited,
        )
        answer = df_records(limited)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
