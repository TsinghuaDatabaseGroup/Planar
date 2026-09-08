#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-087."""

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

TASK_ID = "legal_contracts-087"
QUALIFICATION_FLAGS = (
    "return_or_destroy",
    "injunctive_relief",
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
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
            "SEM_FILTER(exhibit type exactly NDA with identifiable title)",
            input_rows=len(documents),
        ) as step:
            pure_ndas = documents.sem_filter(
                "The document {text} has exhibit type exactly NDA, is not a dual-"
                "labelled exhibit, and has an identifiable agreement title."
            ).reset_index(drop=True)
            step.set_output(pure_ndas)

        with tracker.step(
            "SEM_FILTER(explicit confidential-information definition)",
            input_rows=len(pure_ndas),
        ) as step:
            defined = pure_ndas.sem_filter(
                "The agreement {text} explicitly defines confidential information."
            ).reset_index(drop=True)
            step.set_output(defined)

        with tracker.step(
            "SEM_EXTRACT(title and complete-protection flags)",
            input_rows=len(defined),
        ) as step:
            extracted = defined.sem_extract(
                input_cols=["text"],
                output_cols={
                    "agreement_title": "the identifiable stated agreement title",
                    "return_or_destroy": (
                        "true if return or destruction of confidential materials is "
                        "required"
                    ),
                    "injunctive_relief": (
                        "true if injunctive or equitable relief is provided"
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
                    "third_party_receipt": (
                        "true if an operative lawful third-party-receipt exception is present"
                    ),
                    "legal_compulsion": (
                        "true if an operative legal-compulsion exception is present"
                    ),
                    "consent": (
                        "true if an operative disclosure-with-consent exception is present"
                    ),
                },
            )
            extracted["agreement_title"] = extracted["agreement_title"].map(clean_text)
            for column in QUALIFICATION_FLAGS:
                extracted[column] = extracted[column].map(parse_bool)
            step.set_output(extracted)

        projected = extracted[["agreement_title"]].copy()
        projected["normalized_agreement_title"] = extracted["agreement_title"].map(
            _normalize_title
        )
        projected["qualifies"] = extracted[list(QUALIFICATION_FLAGS)].all(
            axis=1
        ).astype(int)
        projected = projected[
            ["normalized_agreement_title", "qualifies"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(normalized title, complete-profile qualification flag)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("normalized_agreement_title", sort=False)
            .agg(
                family_size=("normalized_agreement_title", "size"),
                qualifying_member_count=("qualifies", "sum"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([normalized title], family size, qualifying-member count)",
            len(projected),
            len(grouped),
            output=grouped,
        )

        qualifying_families = grouped.loc[
            (grouped["family_size"] >= 5)
            & (grouped["qualifying_member_count"] >= 1)
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(family_size >= 5 AND qualifying_member_count >= 1)",
            len(grouped),
            len(qualifying_families),
            output=qualifying_families,
        )

        ordered = qualifying_families.sort_values(
            [
                "qualifying_member_count",
                "family_size",
                "normalized_agreement_title",
            ],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(qualifying count DESC, family size DESC, title ASC)",
            len(qualifying_families),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
