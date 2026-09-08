#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-091."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    parse_number,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-091"


def _entity_binding(parent_company, subsidiaries):
    return (
        f"parent company: {parent_company}\n"
        f"explicitly listed subsidiaries: {', '.join(subsidiaries)}"
    )


def main():
    setup(max_tokens=16384, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        subsidiary_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS left",
            None,
            len(subsidiary_documents),
            output=subsidiary_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-21 subsidiary list with explicit parent)",
            input_rows=len(subsidiary_documents),
        ) as step:
            subsidiary_filings = subsidiary_documents.sem_filter(
                "The document {text} is an EX-21 subsidiary-list filing that "
                "explicitly identifies its parent company."
            ).reset_index(drop=True)
            step.set_output(subsidiary_filings)

        with tracker.step(
            "SEM_EXTRACT(parent, subsidiary entities, and count)",
            input_rows=len(subsidiary_filings),
        ) as step:
            subsidiary_records = subsidiary_filings.sem_extract(
                input_cols=["text"],
                output_cols={
                    "parent_company": "the explicitly identified parent company",
                    "subsidiary_entities": (
                        "a list of every explicitly named subsidiary entity"
                    ),
                    "subsidiary_count": (
                        "the number of listed subsidiary legal entities as an integer"
                    ),
                },
            )
            subsidiary_records["parent_company"] = subsidiary_records[
                "parent_company"
            ].map(clean_text)
            subsidiary_records["subsidiary_entities"] = subsidiary_records[
                "subsidiary_entities"
            ].map(parse_string_list)
            subsidiary_records["subsidiary_count"] = subsidiary_records[
                "subsidiary_count"
            ].map(parse_number)
            subsidiary_records["ex21_entity_binding"] = [
                _entity_binding(parent, subsidiaries)
                for parent, subsidiaries in zip(
                    subsidiary_records["parent_company"],
                    subsidiary_records["subsidiary_entities"],
                )
            ]
            subsidiary_records = subsidiary_records[
                [
                    "parent_company",
                    "subsidiary_count",
                    "ex21_entity_binding",
                ]
            ].reset_index(drop=True)
            step.set_output(subsidiary_records)

        agreement_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS right",
            None,
            len(agreement_documents),
            output=agreement_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-10 employment, offer, or separation agreement)",
            input_rows=len(agreement_documents),
        ) as step:
            ex10_agreements = agreement_documents.sem_filter(
                "The document {text} is an EX-10 employment, offer, or separation "
                "agreement."
            ).reset_index(drop=True)
            step.set_output(ex10_agreements)

        with tracker.step(
            "SEM_FILTER(non-compete and non-solicitation with stated durations)",
            input_rows=len(ex10_agreements),
        ) as step:
            restrictive_agreements = ex10_agreements.sem_filter(
                "The agreement {text} contains an operative non-compete clause and at "
                "least one operative employee or customer non-solicitation clause, "
                "and states the duration of both matched restrictions."
            ).reset_index(drop=True)
            step.set_output(restrictive_agreements)

        with tracker.step(
            "SEM_EXTRACT(EX-10 employer and company entities)",
            input_rows=len(restrictive_agreements),
        ) as step:
            agreement_records = restrictive_agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "ex10_entities": (
                        "a list of all employer and company entity names stated in "
                        "the agreement"
                    )
                },
            ).rename(columns={"document_id": "ex10_document_id"})
            agreement_records["ex10_entities"] = agreement_records[
                "ex10_entities"
            ].map(parse_string_list)
            agreement_records["ex10_entity_binding"] = agreement_records[
                "ex10_entities"
            ].map(lambda values: ", ".join(values))
            agreement_records = agreement_records[
                ["ex10_document_id", "ex10_entity_binding"]
            ].reset_index(drop=True)
            step.set_output(agreement_records)

        with tracker.step(
            "SEM_JOIN(EX-10 entity is EX-21 parent or listed subsidiary)",
            input_rows={
                "EX21": len(subsidiary_records),
                "EX10": len(agreement_records),
            },
        ) as step:
            if subsidiary_records.empty or agreement_records.empty:
                matched = pd.DataFrame(
                    columns=[
                        "parent_company",
                        "subsidiary_count",
                        "ex10_document_id",
                    ]
                )
            else:
                matched = subsidiary_records.sem_join(
                    agreement_records,
                    "At least one employer or company entity in the EX-10 record "
                    "{ex10_entity_binding:right} is either the parent company or an "
                    "entity explicitly named in the EX-21 record "
                    "{ex21_entity_binding:left}, allowing ordinary capitalization, "
                    "punctuation, and corporate-suffix variation."
                ).reset_index(drop=True)
            step.set_output(matched)

        grouped = (
            matched.groupby("parent_company", sort=False)
            .agg(
                subsidiary_count=("subsidiary_count", "max"),
                qualifying_agreement_count=("ex10_document_id", "nunique"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(parent company, max subsidiary count, distinct EX-10 count)",
            len(matched),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
