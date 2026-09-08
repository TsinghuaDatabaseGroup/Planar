#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-085."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    clean_text,
    df_records,
    load_document_corpus,
    normalize_enum,
    normalize_free_label,
    parse_number,
    parse_optional_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-085"
FILING_FAMILIES = (
    "insider_trading",
    "clawback",
    "auditor_consent",
    "subsidiary_list",
)


def _sorted_distinct_non_null(values):
    return sorted({str(value) for value in values if pd.notna(value) and str(value)})


def _any_true(values):
    return any(value is True for value in values)


def _normalize_optional_label(value):
    return None if clean_optional_text(value) is None else normalize_free_label(value)


def main():
    setup(max_tokens=1024, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        sox_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS SOX",
            None,
            len(sox_documents),
            output=sox_documents,
        )
        with tracker.step(
            "SEM_FILTER(SOX Section 302 certification)",
            input_rows=len(sox_documents),
        ) as step:
            sox_certifications = sox_documents.sem_filter(
                "The document {text} is a genuine Sarbanes-Oxley Section 302 "
                "certification."
            ).reset_index(drop=True)
            step.set_output(sox_certifications)

        with tracker.step(
            "SEM_EXTRACT(certified company and certifying officer)",
            input_rows=len(sox_certifications),
        ) as step:
            sox_records = sox_certifications.sem_extract(
                input_cols=["text"],
                output_cols={
                    "sox_company": "the certified company entity",
                    "certifying_officer": (
                        "the certifying officer's full name together with stated title"
                    ),
                },
            )
            sox_records["sox_company"] = sox_records["sox_company"].map(clean_text)
            sox_records["certifying_officer"] = sox_records[
                "certifying_officer"
            ].map(clean_text)
            step.set_output(sox_records)

        sox_groups = (
            sox_records.groupby("sox_company", sort=False)
            .agg(
                certifying_officers=(
                    "certifying_officer",
                    _sorted_distinct_non_null,
                )
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(SOX company, sorted distinct certifying officers)",
            len(sox_records),
            len(sox_groups),
            output=sox_groups,
        )

        other_documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS OTHER",
            None,
            len(other_documents),
            output=other_documents,
        )
        with tracker.step(
            "SEM_FILTER(insider, clawback, auditor-consent, or subsidiary-list filing)",
            input_rows=len(other_documents),
        ) as step:
            other_filings = other_documents.sem_filter(
                "The document {text} is an insider-trading policy, a clawback policy, "
                "an independent auditor consent letter, or a subsidiary-list filing."
            ).reset_index(drop=True)
            step.set_output(other_filings)

        with tracker.step(
            "SEM_EXTRACT(OTHER filing family and family-specific fields)",
            input_rows=len(other_filings),
        ) as step:
            other_records = other_filings.sem_extract(
                input_cols=["text"],
                output_cols={
                    "other_company": (
                        "the company entity to which the filing applies"
                    ),
                    "filing_family": (
                        "exactly insider_trading, clawback, auditor_consent, or "
                        "subsidiary_list"
                    ),
                    "auditor_firm": (
                        "the auditor firm for an auditor-consent filing, or null when "
                        "not applicable or unstated"
                    ),
                    "misconduct_trigger": (
                        "for a clawback policy, true if misconduct triggers recovery "
                        "and false if it does not; null for other filing families"
                    ),
                    "covered_person_category": (
                        "for a clawback policy, a concise normalized covered-person "
                        "category; null for other filing families or when unstated"
                    ),
                    "subsidiary_count": (
                        "for a subsidiary-list filing, the number of listed subsidiary "
                        "legal entities as an integer; null for other filing families"
                    ),
                },
            )
            other_records["other_company"] = other_records["other_company"].map(
                clean_text
            )
            other_records["filing_family"] = other_records["filing_family"].map(
                lambda value: normalize_enum(value, FILING_FAMILIES)
            )
            other_records["auditor_firm"] = other_records["auditor_firm"].map(
                clean_optional_text
            )
            other_records["misconduct_trigger"] = other_records[
                "misconduct_trigger"
            ].map(parse_optional_bool)
            other_records["covered_person_category"] = other_records[
                "covered_person_category"
            ].map(_normalize_optional_label)
            other_records["subsidiary_count"] = other_records[
                "subsidiary_count"
            ].map(parse_number)
            step.set_output(other_records)

        other_groups = (
            other_records.groupby("other_company", sort=False)
            .agg(
                matched_families=("filing_family", _sorted_distinct_non_null),
                auditor_firms=("auditor_firm", _sorted_distinct_non_null),
                has_misconduct_trigger=("misconduct_trigger", _any_true),
                covered_person_categories=(
                    "covered_person_category",
                    _sorted_distinct_non_null,
                ),
                largest_subsidiary_count=("subsidiary_count", "max"),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(OTHER company, filing-family evidence and aggregates)",
            len(other_records),
            len(other_groups),
            output=other_groups,
        )

        qualifying_other = other_groups.loc[
            other_groups["matched_families"].map(
                lambda values: set(FILING_FAMILIES).issubset(values)
            )
            & other_groups["has_misconduct_trigger"]
            & other_groups["largest_subsidiary_count"].notna()
            & (other_groups["largest_subsidiary_count"] >= 30)
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(all four families, misconduct trigger, subsidiary count >= 30)",
            len(other_groups),
            len(qualifying_other),
            output=qualifying_other,
        )

        with tracker.step(
            "SEM_JOIN(SOX and OTHER on same company entity)",
            input_rows={"SOX": len(sox_groups), "OTHER": len(qualifying_other)},
        ) as step:
            if sox_groups.empty or qualifying_other.empty:
                matched = pd.DataFrame(
                    columns=[
                        "sox_company",
                        "certifying_officers",
                        "auditor_firms",
                        "covered_person_categories",
                        "largest_subsidiary_count",
                    ]
                )
            else:
                matched = sox_groups.sem_join(
                    qualifying_other,
                    "The SOX-certified company {sox_company:left} and the company "
                    "represented by the other filings {other_company:right} are the "
                    "same corporate entity despite naming, capitalization, punctuation, "
                    "or corporate-suffix variation."
                ).reset_index(drop=True)
            step.set_output(matched)

        projected = matched.rename(columns={"sox_company": "company"})[
            [
                "company",
                "certifying_officers",
                "auditor_firms",
                "covered_person_categories",
                "largest_subsidiary_count",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(company and cross-family aggregates)",
            len(matched),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            ["largest_subsidiary_count", "company"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(largest_subsidiary_count DESC, company ASC)",
            len(projected),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
