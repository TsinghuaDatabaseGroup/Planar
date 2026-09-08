#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-096."""

import os
import sys

import pandas as pd

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

TASK_ID = "legal_contracts-096"


def _binding(company, person):
    return f"company: {company}\nperson: {person}"


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        certification_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS CERT906",
            None,
            len(certification_documents),
            output=certification_documents,
        )
        with tracker.step(
            "SEM_FILTER(SOX Section 906 certification exhibit)",
            input_rows=len(certification_documents),
        ) as step:
            certifications = certification_documents.sem_filter(
                "The document {text} is a SOX Section 906 certification exhibit "
                "filed as EX-32."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_EXTRACT(certification company, officer, and document ID)",
            input_rows=len(certifications),
        ) as step:
            certification_records = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "certification_company": (
                        "the company entity covered by the certification"
                    ),
                    "certifying_officer": (
                        "the full name of the officer who signs or certifies it"
                    ),
                },
            ).rename(columns={"document_id": "certification_document_id"})
            certification_records["certification_company"] = certification_records[
                "certification_company"
            ].map(clean_optional_text)
            certification_records["certifying_officer"] = certification_records[
                "certifying_officer"
            ].map(clean_optional_text)
            certification_records = certification_records.dropna(
                subset=["certification_company", "certifying_officer"]
            ).copy()
            certification_records["certification_binding"] = (
                certification_records.apply(
                    lambda row: _binding(
                        row["certification_company"], row["certifying_officer"]
                    ),
                    axis=1,
                )
            )
            certification_records = certification_records[
                [
                    "certification_document_id",
                    "certification_company",
                    "certifying_officer",
                    "certification_binding",
                ]
            ].reset_index(drop=True)
            step.set_output(certification_records)

        agreement_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS AGREEMENTS",
            None,
            len(agreement_documents),
            output=agreement_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-10 named employee and explicit non-compete duration)",
            input_rows=len(agreement_documents),
        ) as step:
            agreements = agreement_documents.sem_filter(
                "The document {text} is an EX-10 employment or restrictive-covenant "
                "agreement that names the employee and explicitly states a "
                "non-compete duration."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(agreement company, employee, document, and duration)",
            input_rows=len(agreements),
        ) as step:
            agreement_records = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "agreement_company": (
                        "the company or employer entity that is party to the agreement"
                    ),
                    "employee_name": "the full name of the employee",
                    "non_compete_duration_years": (
                        "the explicitly stated non-compete duration converted to years"
                    ),
                },
            ).rename(columns={"document_id": "ex10_document"})
            agreement_records["agreement_company"] = agreement_records[
                "agreement_company"
            ].map(clean_optional_text)
            agreement_records["employee_name"] = agreement_records[
                "employee_name"
            ].map(clean_optional_text)
            agreement_records["non_compete_duration_years"] = agreement_records[
                "non_compete_duration_years"
            ].map(parse_number)
            agreement_records = agreement_records.dropna(
                subset=[
                    "agreement_company",
                    "employee_name",
                    "non_compete_duration_years",
                ]
            ).copy()
            agreement_records["agreement_binding"] = agreement_records.apply(
                lambda row: _binding(row["agreement_company"], row["employee_name"]),
                axis=1,
            )
            agreement_records = agreement_records[
                [
                    "ex10_document",
                    "agreement_company",
                    "employee_name",
                    "non_compete_duration_years",
                    "agreement_binding",
                ]
            ].reset_index(drop=True)
            step.set_output(agreement_records)

        with tracker.step(
            "SEM_JOIN(same company and same individual)",
            input_rows={
                "CERT906": len(certification_records),
                "AGREEMENTS": len(agreement_records),
            },
        ) as step:
            if certification_records.empty or agreement_records.empty:
                matched = pd.DataFrame(
                    columns=[
                        "certification_document_id",
                        "certification_company",
                        "certifying_officer",
                        "ex10_document",
                        "non_compete_duration_years",
                    ]
                )
            else:
                matched = certification_records.sem_join(
                    agreement_records,
                    "The certification record {certification_binding:left} and the "
                    "agreement record {agreement_binding:right} concern the same "
                    "company, and their named people are the same individual. Allow "
                    "clear middle-name, nickname, initial, and shortened-name variants, "
                    "but require both the company match and person match."
                ).reset_index(drop=True)
            step.set_output(matched)

        grouped = (
            matched.groupby(
                [
                    "certification_company",
                    "certifying_officer",
                    "ex10_document",
                    "non_compete_duration_years",
                ],
                sort=False,
                dropna=False,
            )["certification_document_id"]
            .nunique()
            .rename("certification_count")
            .reset_index()
            .rename(columns={"certification_company": "company"})
        )
        grouped = grouped[
            [
                "company",
                "certifying_officer",
                "ex10_document",
                "certification_count",
                "non_compete_duration_years",
            ]
        ]
        tracker.record(
            "GROUP_BY(officer-agreement pair, distinct certification count)",
            len(matched),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
