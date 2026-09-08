#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-076."""

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
    normalize_enum,
    normalize_iso_date,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-076"
PERIOD_TYPES = ("annual", "quarterly", "other")
JOIN_KEYS = ["company_key", "reporting_period_type", "reporting_period_end_date"]


def _lexicographic_min(values):
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    return min(cleaned) if cleaned else None


def _sorted_values(values):
    return sorted({str(value) for value in values if pd.notna(value)})


def _sorted_distinct_flatten(values):
    flattened = {
        str(item)
        for value in values
        for item in value
        if item is not None and str(item)
    }
    return sorted(flattened)


def _extract_certifications(documents, section, tracker):
    with tracker.step(
        f"SEM_FILTER(SOX Section {section} certification)",
        input_rows=len(documents),
    ) as step:
        certifications = documents.sem_filter(
            f"The document {{text}} is a genuine Sarbanes-Oxley Section {section} "
            "certification."
        ).reset_index(drop=True)
        step.set_output(certifications)

    with tracker.step(
        f"SEM_EXTRACT(Section {section} company, period, and officers)",
        input_rows=len(certifications),
    ) as step:
        extracted = certifications.sem_extract(
            input_cols=["text"],
            output_cols={
                "company_key": (
                    "a stable normalized identity key for the stated company, using "
                    "lowercase canonical company identity rather than a display-name "
                    "variant"
                ),
                "company_name_variant": (
                    "the company name exactly as stated in this certification"
                ),
                "reporting_period_type": (
                    "the explicitly stated reporting-period type as annual, "
                    "quarterly, or other; return null when unstated"
                ),
                "reporting_period_end_date": (
                    "the explicitly stated reporting-period end date formatted as "
                    "YYYY-MM-DD; return null when unstated"
                ),
                "certifying_officers": (
                    "a list of the full names of the officers who certify the document"
                ),
            },
        )
        extracted["company_key"] = extracted["company_key"].map(clean_optional_text)
        extracted["company_name_variant"] = extracted[
            "company_name_variant"
        ].map(clean_optional_text)
        extracted["reporting_period_type"] = extracted[
            "reporting_period_type"
        ].map(lambda value: normalize_enum(value, PERIOD_TYPES))
        extracted["reporting_period_end_date"] = extracted[
            "reporting_period_end_date"
        ].map(normalize_iso_date)
        extracted["certifying_officers"] = extracted[
            "certifying_officers"
        ].map(parse_string_list)
        step.set_output(extracted)

    dated = extracted.loc[
        extracted["reporting_period_type"].notna()
        & extracted["reporting_period_end_date"].notna()
    ].reset_index(drop=True)
    tracker.record(
        f"FILTER(Section {section} period type/date IS NOT NULL)",
        len(extracted),
        len(dated),
        output=dated,
    )

    grouped = (
        dated.groupby(JOIN_KEYS, sort=False, dropna=False)
        .agg(
            company_name_variant=("company_name_variant", _lexicographic_min),
            document_ids=("document_id", _sorted_values),
            certifying_officers=(
                "certifying_officers",
                _sorted_distinct_flatten,
            ),
        )
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            "company_name_variant": f"company_name_{section}",
            "document_ids": f"section_{section}_document_ids",
            "certifying_officers": f"section_{section}_certifying_officers",
        }
    )
    tracker.record(
        f"GROUP_BY(Section {section}, company and reporting period)",
        len(dated),
        len(grouped),
        output=grouped,
    )
    return grouped


def main():
    setup(max_tokens=1024, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        section_302_docs = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS section_302_docs",
            None,
            len(section_302_docs),
            output=section_302_docs,
        )
        section_302_groups = _extract_certifications(
            section_302_docs,
            "302",
            tracker,
        )

        section_906_docs = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS section_906_docs",
            None,
            len(section_906_docs),
            output=section_906_docs,
        )
        section_906_groups = _extract_certifications(
            section_906_docs,
            "906",
            tracker,
        )

        joined = section_302_groups.dropna(subset=["company_key"]).merge(
            section_906_groups.dropna(subset=["company_key"]),
            on=JOIN_KEYS,
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(section_302_groups, section_906_groups, on=company/period)",
            {
                "section_302_groups": len(section_302_groups),
                "section_906_groups": len(section_906_groups),
            },
            len(joined),
            output=joined,
        )

        projected = joined.copy()
        projected["company_name"] = projected.apply(
            lambda row: _lexicographic_min(
                [row["company_name_302"], row["company_name_906"]]
            ),
            axis=1,
        )
        projected = projected[
            [
                "company_name",
                "reporting_period_type",
                "reporting_period_end_date",
                "section_302_document_ids",
                "section_302_certifying_officers",
                "section_906_document_ids",
                "section_906_certifying_officers",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(company name, period, and Section 302/906 details)",
            len(joined),
            len(projected),
            output=projected,
        )

        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
