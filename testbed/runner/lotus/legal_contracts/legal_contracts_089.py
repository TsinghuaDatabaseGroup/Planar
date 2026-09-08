#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-089."""

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

TASK_ID = "legal_contracts-089"
EXCEPTION_COLUMNS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
)
BUCKET_ORDER = ("high_5_to_6", "medium_3_to_4", "low_0_to_2")


def _normalize_title(value):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def _coverage_bucket(value):
    if value >= 5:
        return "high_5_to_6"
    if value >= 3:
        return "medium_3_to_4"
    return "low_0_to_2"


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
            "SEM_EXTRACT(title and six exception flags)",
            input_rows=len(defined),
        ) as step:
            extracted = defined.sem_extract(
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
            for column in EXCEPTION_COLUMNS:
                extracted[column] = extracted[column].map(parse_bool)
            step.set_output(extracted)

        projected = extracted[["agreement_title"]].copy()
        projected["normalized_agreement_title"] = extracted["agreement_title"].map(
            _normalize_title
        )
        projected["exception_count"] = extracted[list(EXCEPTION_COLUMNS)].sum(axis=1)
        projected = projected[
            ["normalized_agreement_title", "exception_count"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(normalized title, exception count)",
            len(extracted),
            len(projected),
            output=projected,
        )

        title_families = (
            projected.groupby("normalized_agreement_title", sort=False)
            .agg(
                family_size=("normalized_agreement_title", "size"),
                average_exception_count=("exception_count", "mean"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([normalized title], family size, average exception count)",
            len(projected),
            len(title_families),
            output=title_families,
        )

        large_families = title_families.loc[
            title_families["family_size"] >= 3
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(family_size >= 3)",
            len(title_families),
            len(large_families),
            output=large_families,
        )

        bucketed = large_families[["family_size"]].copy()
        bucketed["coverage_bucket"] = large_families[
            "average_exception_count"
        ].map(_coverage_bucket)
        bucketed = bucketed[["coverage_bucket", "family_size"]].reset_index(
            drop=True
        )
        tracker.record(
            "PROJECT(coverage bucket, family size)",
            len(large_families),
            len(bucketed),
            output=bucketed,
        )

        grouped = (
            bucketed.groupby("coverage_bucket", sort=False)
            .agg(
                family_count=("coverage_bucket", "size"),
                average_family_size=("family_size", "mean"),
            )
            .reset_index()
        )
        grouped["average_family_size"] = grouped["average_family_size"].round(2)
        tracker.record(
            "GROUP_BY([coverage_bucket], family count, average family size)",
            len(bucketed),
            len(grouped),
            output=grouped,
        )

        bucket_order = {
            bucket: index for index, bucket in enumerate(BUCKET_ORDER)
        }
        ordered = grouped.sort_values(
            "coverage_bucket",
            key=lambda values: values.map(bucket_order),
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(explicit coverage-bucket order)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
