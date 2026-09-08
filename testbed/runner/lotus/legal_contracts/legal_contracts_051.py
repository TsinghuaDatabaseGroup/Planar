#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-051."""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    parse_record_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-051"
TARGET_JURISDICTIONS = {
    "cayman islands",
    "british virgin islands",
    "mauritius",
    "luxembourg",
}


def _normalize_jurisdiction(value):
    jurisdiction = clean_optional_text(value)
    if jurisdiction is None:
        return None
    normalized = jurisdiction.casefold().replace("the ", "", 1).strip()
    if normalized in {"bvi", "british virgin islands"}:
        return "British Virgin Islands"
    if normalized == "cayman islands":
        return "Cayman Islands"
    if normalized == "mauritius":
        return "Mauritius"
    if normalized == "luxembourg":
        return "Luxembourg"
    return jurisdiction


def _normalize_subsidiaries(value):
    subsidiaries = []
    for record in parse_record_list(value):
        subsidiary = dict(record)
        subsidiary["jurisdiction"] = _normalize_jurisdiction(
            subsidiary.get("jurisdiction")
        )
        subsidiaries.append(subsidiary)
    return subsidiaries


def _contains_target(subsidiaries):
    return any(
        str(subsidiary.get("jurisdiction", "")).casefold()
        in TARGET_JURISDICTIONS
        for subsidiary in subsidiaries
    )


def _project_row(row):
    jurisdictions = [
        subsidiary.get("jurisdiction")
        for subsidiary in row["subsidiaries"]
        if subsidiary.get("jurisdiction") is not None
    ]
    counts = Counter(jurisdictions)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    return {
        "document_id": row["document_id"],
        "parent_company": row["parent_company"],
        "subsidiary_count": len(row["subsidiaries"]),
        "distinct_jurisdiction_count": len(counts),
        "top_jurisdictions": [
            {"jurisdiction": jurisdiction, "subsidiary_count": count}
            for jurisdiction, count in ordered
        ],
    }


def main():
    setup(max_tokens=16384, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(genuine subsidiary-list filing)",
            input_rows=len(documents),
        ) as step:
            filings = documents.sem_filter(
                "The document {text} is a genuine subsidiary-list filing."
            ).reset_index(drop=True)
            step.set_output(filings)

        with tracker.step(
            "SEM_EXTRACT(parent company and listed subsidiaries)",
            input_rows=len(filings),
        ) as step:
            extracted = filings.sem_extract(
                input_cols=["text"],
                output_cols={
                    "parent_company": (
                        "the parent company identified by the filing, or null when "
                        "none is identified"
                    ),
                    "subsidiaries": (
                        "a JSON list containing one object for every listed subsidiary; "
                        "each object must have the subsidiary name and its stated "
                        "incorporation jurisdiction, using null when that jurisdiction "
                        "is not stated"
                    ),
                },
            )
            extracted["parent_company"] = extracted["parent_company"].map(
                clean_optional_text
            )
            extracted["subsidiaries"] = extracted["subsidiaries"].map(
                _normalize_subsidiaries
            )
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["parent_company"].notna()
            & extracted["subsidiaries"].map(_contains_target)
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(parent_company IS NOT NULL AND ANY_IN(target jurisdictions))",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        projected = df_records(qualifying)
        projected_records = [_project_row(row) for row in projected]
        tracker.record(
            "PROJECT(document, parent, subsidiary count, distinct jurisdictions, top 3 jurisdictions)",
            len(qualifying),
            len(projected_records),
            output=projected_records,
        )
        answer = projected_records

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
