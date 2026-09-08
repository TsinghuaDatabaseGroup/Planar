#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-101."""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    normalize_company_name,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-101"


def _normalize_person_key(value):
    text = clean_optional_text(value)
    if text is None:
        return None
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split()) or None


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        employment_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS EMPLOYMENT",
            None,
            len(employment_documents),
            output=employment_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-10 employment with finite employee non-solicitation)",
            input_rows=len(employment_documents),
        ) as step:
            employment = employment_documents.sem_filter(
                "The document {text} is an EX-10 employment agreement with a finite, "
                "expressly stated employee non-solicitation period."
            ).reset_index(drop=True)
            step.set_output(employment)

        with tracker.step(
            "SEM_EXTRACT(employer, employee, person key, and duration)",
            input_rows=len(employment),
        ) as step:
            employment_records = employment.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_name": "the employer name stated in the agreement",
                    "company_key": (
                        "the employer as a concise canonical company name"
                    ),
                    "employee_name": "the full name of the employee",
                    "person_key": (
                        "the employee as a normalized person key that resolves clear "
                        "middle-name, initial, nickname, and shortened-name variants"
                    ),
                    "non_solicitation_duration_years": (
                        "the finite employee non-solicitation period converted to years"
                    ),
                },
            ).rename(columns={"document_id": "employment_document"})
            employment_records["company_name"] = employment_records[
                "company_name"
            ].map(clean_optional_text)
            employment_records["company_key"] = employment_records[
                "company_key"
            ].map(normalize_company_name)
            employment_records["employee_name"] = employment_records[
                "employee_name"
            ].map(clean_optional_text)
            employment_records["person_key"] = employment_records["person_key"].map(
                _normalize_person_key
            )
            employment_records["non_solicitation_duration_years"] = (
                employment_records["non_solicitation_duration_years"].map(parse_number)
            )
            employment_records = employment_records.dropna(
                subset=[
                    "company_name",
                    "company_key",
                    "employee_name",
                    "person_key",
                    "non_solicitation_duration_years",
                ]
            )[
                [
                    "employment_document",
                    "company_name",
                    "company_key",
                    "employee_name",
                    "person_key",
                    "non_solicitation_duration_years",
                ]
            ].reset_index(drop=True)
            step.set_output(employment_records)

        certification_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS CERT302",
            None,
            len(certification_documents),
            output=certification_documents,
        )
        with tracker.step(
            "SEM_FILTER(EX-31 SOX Section 302 certification)",
            input_rows=len(certification_documents),
        ) as step:
            certifications = certification_documents.sem_filter(
                "The document {text} is an EX-31 SOX Section 302 certification."
            ).reset_index(drop=True)
            step.set_output(certifications)

        with tracker.step(
            "SEM_EXTRACT(certified company, officer, person key, and title)",
            input_rows=len(certifications),
        ) as step:
            certification_records = certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the certified company as a concise canonical company name"
                    ),
                    "officer_name": "the full name of the certifying officer",
                    "person_key": (
                        "the officer as a normalized person key that resolves clear "
                        "middle-name, initial, nickname, and shortened-name variants"
                    ),
                    "officer_title": "the officer title stated in the certification",
                },
            )
            certification_records["company_key"] = certification_records[
                "company_key"
            ].map(normalize_company_name)
            certification_records["officer_name"] = certification_records[
                "officer_name"
            ].map(clean_optional_text)
            certification_records["person_key"] = certification_records[
                "person_key"
            ].map(_normalize_person_key)
            certification_records["officer_title"] = certification_records[
                "officer_title"
            ].map(clean_optional_text)
            certification_records = certification_records.dropna(
                subset=["company_key", "officer_name", "person_key", "officer_title"]
            )[
                ["company_key", "officer_name", "person_key", "officer_title"]
            ].reset_index(drop=True)
            step.set_output(certification_records)

        policy_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS POLICIES",
            None,
            len(policy_documents),
            output=policy_documents,
        )
        with tracker.step(
            "SEM_FILTER(document is an EX-19 insider trading policy)",
            input_rows=len(policy_documents),
        ) as step:
            policies = policy_documents.sem_filter(
                "The document {text} is an EX-19 insider trading policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(policy company key and covered-person scope)",
            input_rows=len(policies),
        ) as step:
            policy_records = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_key": (
                        "the company governed by the policy, as a concise canonical "
                        "company name"
                    ),
                    "covered_person_scope": (
                        "the policy's stated categories, roles, or titles of covered "
                        "people"
                    ),
                },
            )
            policy_records["company_key"] = policy_records["company_key"].map(
                normalize_company_name
            )
            policy_records["covered_person_scope"] = policy_records[
                "covered_person_scope"
            ].map(clean_optional_text)
            policy_records = policy_records.dropna(
                subset=["company_key", "covered_person_scope"]
            )[
                ["company_key", "covered_person_scope"]
            ].reset_index(drop=True)
            step.set_output(policy_records)

        employee_officers = employment_records.merge(
            certification_records,
            on=["company_key", "person_key"],
            how="inner",
        )
        tracker.record(
            "JOIN(EMPLOYMENT, CERT302, on company_key and person_key)",
            {
                "EMPLOYMENT": len(employment_records),
                "CERT302": len(certification_records),
            },
            len(employee_officers),
            output=employee_officers,
        )

        matched = employee_officers.merge(
            policy_records,
            on="company_key",
            how="inner",
        )
        tracker.record(
            "JOIN(MATCHED.company_key = POLICIES.company_key)",
            {
                "EMPLOYMENT_CERT302": len(employee_officers),
                "POLICIES": len(policy_records),
            },
            len(matched),
            output=matched,
        )

        with tracker.step(
            "SEM_FILTER(officer falls within covered-person scope)",
            input_rows=len(matched),
        ) as step:
            if matched.empty:
                covered = matched.copy()
            else:
                covered = matched.sem_filter(
                    "The certifying officer title {officer_title} falls within the "
                    "insider trading policy's stated covered-person scope "
                    "{covered_person_scope}."
                ).reset_index(drop=True)
            step.set_output(covered)

        projected = covered[
            [
                "company_name",
                "officer_name",
                "employment_document",
                "non_solicitation_duration_years",
            ]
        ].rename(
            columns={"company_name": "company", "officer_name": "individual"}
        )
        tracker.record(
            "PROJECT(company, individual, employment document, duration)",
            len(covered),
            len(projected),
            output=projected,
        )

        deduplicated = projected.drop_duplicates(
            subset=[
                "company",
                "individual",
                "employment_document",
                "non_solicitation_duration_years",
            ]
        ).reset_index(drop=True)
        tracker.record(
            "DEDUP(company, individual, document, duration)",
            len(projected),
            len(deduplicated),
            output=deduplicated,
        )
        answer = df_records(deduplicated)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
