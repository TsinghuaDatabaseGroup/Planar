#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-093."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_mixed_documents,
    memory_dataset,
    normalize_scalar_value,
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-093"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def entity_person_binding(company: str | None, person: str | None) -> str:
    return f"company: {company or ''}\nperson: {person or ''}"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        certification_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, certification_documents)

        certification_filter_plan = memory_dataset(
            f"{TASK_ID}-certifications", certification_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is a SOX Section 302 officer "
                "certification exhibit filed as EX-31."
            ),
            depends_on=["text"],
        )
        started = time.time()
        certification_filter_result = certification_filter_plan.run(config)
        certifications = result_frame(
            certification_filter_result, certification_documents
        )
        tracker.record_semantic(
            "sem_filter", len(certification_documents), certifications,
            certification_filter_result, time.time() - started,
        )

        certification_extraction_plan = memory_dataset(
            f"{TASK_ID}-certification-extraction", certifications
        ).sem_map(
            cols=[
                {"name": "certification_company", "type": str,
                 "desc": "The company entity covered by the certification."},
                {"name": "certifying_officer", "type": str,
                 "desc": "The full name of the officer who signs or certifies the document."},
            ],
            desc="Extract the certification company and certifying officer.",
            depends_on=["text"],
        )
        started = time.time()
        certification_extraction_result = certification_extraction_plan.run(config)
        certification_records = result_frame(
            certification_extraction_result,
            certifications,
            ["certification_company", "certifying_officer"],
        )
        certification_records["certification_company"] = certification_records[
            "certification_company"
        ].map(normalize_text)
        certification_records["certifying_officer"] = certification_records[
            "certifying_officer"
        ].map(normalize_text)
        tracker.record_semantic(
            "sem_map", len(certifications), certification_records,
            certification_extraction_result, time.time() - started,
        )

        certification_projected = certification_records.dropna(
            subset=["certification_company", "certifying_officer"]
        )[["certification_company", "certifying_officer"]].copy()
        certification_projected["certification_binding"] = certification_projected.apply(
            lambda row: entity_person_binding(
                row["certification_company"], row["certifying_officer"]
            ),
            axis=1,
        )
        certification_projected = certification_projected.reset_index(drop=True)
        tracker.record("project", len(certification_records), certification_projected)

        agreement_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, agreement_documents)

        agreement_filter_plan = memory_dataset(
            f"{TASK_ID}-agreements", agreement_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-10 employment or restrictive-"
                "covenant agreement that names the employee and explicitly states a "
                "non-compete duration."
            ),
            depends_on=["text"],
        )
        started = time.time()
        agreement_filter_result = agreement_filter_plan.run(config)
        agreements = result_frame(agreement_filter_result, agreement_documents)
        tracker.record_semantic(
            "sem_filter", len(agreement_documents), agreements,
            agreement_filter_result, time.time() - started,
        )

        agreement_extraction_plan = memory_dataset(
            f"{TASK_ID}-agreement-extraction", agreements
        ).sem_map(
            cols=[
                {"name": "agreement_company", "type": str,
                 "desc": "The company or employer entity that is party to the agreement."},
                {"name": "employee_name", "type": str,
                 "desc": "The full name of the employee named by the agreement."},
                {"name": "non_compete_duration_years", "type": float,
                 "desc": "The explicitly stated non-compete duration normalized to years."},
            ],
            desc="Extract the agreement company, employee, and non-compete duration.",
            depends_on=["text"],
        )
        started = time.time()
        agreement_extraction_result = agreement_extraction_plan.run(config)
        agreement_records = result_frame(
            agreement_extraction_result,
            agreements,
            ["agreement_company", "employee_name", "non_compete_duration_years"],
        )
        agreement_records["agreement_company"] = agreement_records[
            "agreement_company"
        ].map(normalize_text)
        agreement_records["employee_name"] = agreement_records["employee_name"].map(
            normalize_text
        )
        agreement_records["non_compete_duration_years"] = agreement_records[
            "non_compete_duration_years"
        ].map(normalize_number)
        tracker.record_semantic(
            "sem_map", len(agreements), agreement_records,
            agreement_extraction_result, time.time() - started,
        )

        agreement_projected = agreement_records.dropna(
            subset=["agreement_company", "employee_name", "non_compete_duration_years"]
        ).copy()
        agreement_projected = agreement_projected.rename(
            columns={"document_id": "ex10_document"}
        )
        agreement_projected["agreement_binding"] = agreement_projected.apply(
            lambda row: entity_person_binding(
                row["agreement_company"], row["employee_name"]
            ),
            axis=1,
        )
        agreement_projected = agreement_projected[
            [
                "ex10_document",
                "agreement_company",
                "employee_name",
                "non_compete_duration_years",
                "agreement_binding",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(agreement_records), agreement_projected)

        join_columns = [
            "certification_company",
            "certifying_officer",
            "certification_binding",
            "ex10_document",
            "agreement_company",
            "employee_name",
            "non_compete_duration_years",
            "agreement_binding",
        ]
        if certification_projected.empty or agreement_projected.empty:
            matched = pd.DataFrame(columns=join_columns)
            tracker.record(
                "sem_join",
                {
                    "left": len(certification_projected),
                    "right": len(agreement_projected),
                },
                matched,
            )
        else:
            join_plan = memory_dataset(
                f"{TASK_ID}-certification-records", certification_projected
            ).sem_join(
                memory_dataset(f"{TASK_ID}-agreement-records", agreement_projected),
                condition=(
                    "The certification and agreement concern the same company, and "
                    "the certifying officer and named employee are the same individual. "
                    "Allow clear middle-name, nickname, initial, and shortened-name "
                    "variants, but require both the company and person to match."
                ),
                depends_on=["certification_binding", "agreement_binding"],
            )
            started = time.time()
            join_result = join_plan.run(config)
            matched = result_frame(join_result)
            if matched.empty:
                matched = pd.DataFrame(columns=join_columns)
            tracker.record_semantic(
                "sem_join",
                {
                    "left": len(certification_projected),
                    "right": len(agreement_projected),
                },
                matched,
                join_result,
                time.time() - started,
            )

        projected = matched[
            [
                "certification_company",
                "certifying_officer",
                "ex10_document",
                "non_compete_duration_years",
            ]
        ].rename(columns={"certification_company": "company"})
        projected = projected.reset_index(drop=True)
        tracker.record("project", len(matched), projected)

        deduplicated = projected.drop_duplicates(
            subset=["company", "certifying_officer", "ex10_document"],
            ignore_index=True,
        )
        tracker.record("dedup", len(projected), deduplicated)
        answer = df_records(deduplicated)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
