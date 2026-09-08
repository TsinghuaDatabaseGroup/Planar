#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-004."""

import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    load_document_corpus,
    parse_bool,
    parse_number,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-004"


def _deduplicate_jurisdictions(value):
    output = []
    seen = set()
    for jurisdiction in parse_string_list(value):
        cleaned = clean_text(jurisdiction)
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _native_number(value):
    if pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def main():
    setup(max_tokens=4096, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(genuine Exhibit 21 subsidiary-list filing)",
            input_rows=len(documents),
        ) as step:
            subsidiary_filings = documents.sem_filter(
                "The document {text} is a genuine Exhibit 21 subsidiary-list filing. "
                "Exclude Exhibit 22 guarantor or co-issuer lists and narrative "
                "documents that merely mention subsidiaries."
            ).reset_index(drop=True)
            step.set_output(subsidiary_filings)

        with tracker.step(
            "SEM_EXTRACT(subsidiary count, jurisdictions, and offshore presence)",
            input_rows=len(subsidiary_filings),
        ) as step:
            extracted = subsidiary_filings.sem_extract(
                input_cols=["text"],
                output_cols={
                    "subsidiary_count": (
                        "the number of distinct subsidiary legal entities listed in "
                        "the filing, as an integer"
                    ),
                    "jurisdictions": (
                        "a deduplicated list of all expressly listed incorporation or "
                        "organization jurisdictions, using conventional full English "
                        "jurisdiction names"
                    ),
                    "has_target_offshore_jurisdiction": (
                        "true when the deduplicated jurisdiction set contains the "
                        "Cayman Islands, the British Virgin Islands (BVI), or "
                        "Luxembourg; false otherwise"
                    ),
                },
            )
            extracted["subsidiary_count"] = extracted["subsidiary_count"].map(
                parse_number
            )
            extracted["jurisdictions"] = extracted["jurisdictions"].map(
                _deduplicate_jurisdictions
            )
            extracted["has_target_offshore_jurisdiction"] = extracted[
                "has_target_offshore_jurisdiction"
            ].map(parse_bool)
            step.set_output(extracted)

        jurisdiction_counts = Counter()
        jurisdiction_names = {}
        for jurisdictions in extracted["jurisdictions"]:
            for jurisdiction in jurisdictions:
                key = jurisdiction.casefold()
                jurisdiction_counts[key] += 1
                jurisdiction_names.setdefault(key, jurisdiction)

        maximum_filing_count = max(jurisdiction_counts.values(), default=0)
        most_prevalent = sorted(
            (
                jurisdiction_names[key]
                for key, count in jurisdiction_counts.items()
                if count == maximum_filing_count
            ),
            key=lambda value: value.casefold(),
        )
        subsidiary_counts = pd.to_numeric(
            extracted["subsidiary_count"], errors="coerce"
        ).dropna()
        offshore_percentage = (
            round(
                100.0
                * float(extracted["has_target_offshore_jurisdiction"].mean()),
                1,
            )
            if len(extracted)
            else 0.0
        )
        answer = {
            "most_prevalent_jurisdictions": most_prevalent,
            "jurisdiction_filing_count": int(maximum_filing_count),
            "min_subsidiary_count": (
                _native_number(subsidiary_counts.min())
                if not subsidiary_counts.empty
                else None
            ),
            "median_subsidiary_count": (
                _native_number(subsidiary_counts.median())
                if not subsidiary_counts.empty
                else None
            ),
            "max_subsidiary_count": (
                _native_number(subsidiary_counts.max())
                if not subsidiary_counts.empty
                else None
            ),
            "offshore_filing_percentage": offshore_percentage,
        }
        tracker.record(
            "GROUP_BY([], jurisdiction frequencies and subsidiary statistics)",
            len(extracted),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
