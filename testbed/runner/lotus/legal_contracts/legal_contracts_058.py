#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-058."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-058"
DIRECTIONS = ("mutual", "unilateral", "unclear")
EXCEPTION_COLUMNS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "lawful_third_party_receipt",
    "legally_compelled_disclosure",
    "disclosure_with_consent",
)


def _recognized_exceptions(row):
    return tuple(column for column in EXCEPTION_COLUMNS if row[column])


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
            "SEM_FILTER(NDA or confidentiality-and-standstill agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(direction and six confidentiality exceptions)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "non_disclosure_direction": (
                        "exactly mutual, unilateral, or unclear according to who owes "
                        "the nondisclosure duty"
                    ),
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
                    "legally_compelled_disclosure": (
                        "true if an operative legally-compelled-disclosure exception "
                        "is present"
                    ),
                    "disclosure_with_consent": (
                        "true if an operative disclosure-with-consent exception is present"
                    ),
                },
            )
            extracted["non_disclosure_direction"] = extracted[
                "non_disclosure_direction"
            ].map(lambda value: normalize_enum(value, DIRECTIONS))
            for column in EXCEPTION_COLUMNS:
                extracted[column] = extracted[column].map(parse_bool)
            step.set_output(extracted)

        projected = extracted[["non_disclosure_direction"]].copy()
        projected["recognized_exceptions"] = extracted.apply(
            _recognized_exceptions,
            axis=1,
        )
        tracker.record(
            "PROJECT(direction, CANONICAL_TRUE_LABELS AS recognized_exceptions)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby(
                ["non_disclosure_direction", "recognized_exceptions"],
                sort=False,
                dropna=False,
            )
            .size()
            .rename("agreement_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([direction, recognized_exceptions], COUNT(*))",
            len(projected),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["agreement_count", "non_disclosure_direction", "recognized_exceptions"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        ordered["recognized_exceptions"] = ordered["recognized_exceptions"].map(list)
        tracker.record(
            "ORDER_BY(agreement_count DESC, direction ASC, exceptions ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record(
            "LIMIT(5)",
            len(ordered),
            len(limited),
            output=limited,
        )
        answer = df_records(limited)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
