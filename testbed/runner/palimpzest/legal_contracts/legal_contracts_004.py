#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-004."""

from __future__ import annotations

import ast
import json
import os
import sys
import time
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_mixed_documents,
    memory_dataset,
    normalize_scalar_value,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-004"
DATASET = "contract-exhibit"


def normalize_string_list(value) -> list[str]:
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return []
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
    if pd.api.types.is_list_like(parsed) and not isinstance(parsed, dict):
        parsed = list(parsed)
    else:
        parsed = [parsed]
    output = []
    seen = set()
    for item in parsed:
        normalized = " ".join(str(item).strip().split())
        if not normalized:
            continue
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def native_number(value):
    value = normalize_scalar_value(value)
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        filing_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a genuine Exhibit 21 "
                "subsidiary-list filing. Exclude Exhibit 22 guarantor or "
                "co-issuer lists and narrative documents that merely mention "
                "subsidiaries."
            ),
            depends_on=["text"],
        )
        started = time.time()
        filing_result = filing_plan.run(config)
        filings = result_frame(filing_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            filings,
            filing_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", filings
        ).sem_map(
            cols=[
                {
                    "name": "subsidiary_count",
                    "type": int | None,
                    "desc": (
                        "The number of distinct subsidiary legal entities "
                        "listed in the filing."
                    ),
                },
                {
                    "name": "jurisdictions",
                    "type": list[str],
                    "desc": (
                        "A deduplicated list of all expressly listed "
                        "incorporation or organization jurisdictions, using "
                        "conventional full English jurisdiction names."
                    ),
                },
                {
                    "name": "has_target_offshore_jurisdiction",
                    "type": bool,
                    "desc": (
                        "True when the deduplicated jurisdiction set contains "
                        "the Cayman Islands, British Virgin Islands (BVI), or "
                        "Luxembourg; false otherwise."
                    ),
                },
            ],
            desc=(
                "Extract the subsidiary count, deduplicated jurisdictions, and "
                "whether a target offshore jurisdiction occurs."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            filings,
            [
                "subsidiary_count",
                "jurisdictions",
                "has_target_offshore_jurisdiction",
            ],
        )
        extracted["subsidiary_count"] = extracted["subsidiary_count"].map(
            normalize_number
        )
        extracted["jurisdictions"] = extracted["jurisdictions"].map(
            normalize_string_list
        )
        extracted["has_target_offshore_jurisdiction"] = extracted[
            "has_target_offshore_jurisdiction"
        ].map(parse_bool)
        tracker.record_semantic(
            "sem_map",
            len(filings),
            extracted,
            extraction_result,
            time.time() - started,
        )

        jurisdiction_counts = Counter()
        jurisdiction_names = {}
        for jurisdictions in extracted["jurisdictions"]:
            for jurisdiction in jurisdictions:
                key = jurisdiction.casefold()
                jurisdiction_counts[key] += 1
                jurisdiction_names.setdefault(key, jurisdiction)

        maximum_filing_count = max(jurisdiction_counts.values(), default=0)
        subsidiary_counts = pd.to_numeric(
            extracted["subsidiary_count"], errors="coerce"
        ).dropna()
        answer = {
            "most_prevalent_jurisdictions": sorted(
                (
                    jurisdiction_names[key]
                    for key, count in jurisdiction_counts.items()
                    if count == maximum_filing_count
                ),
                key=str.casefold,
            ),
            "jurisdiction_filing_count": int(maximum_filing_count),
            "min_subsidiary_count": (
                native_number(subsidiary_counts.min())
                if not subsidiary_counts.empty
                else None
            ),
            "median_subsidiary_count": (
                native_number(subsidiary_counts.median())
                if not subsidiary_counts.empty
                else None
            ),
            "max_subsidiary_count": (
                native_number(subsidiary_counts.max())
                if not subsidiary_counts.empty
                else None
            ),
            "offshore_filing_percentage": (
                round(
                    100.0
                    * float(
                        extracted["has_target_offshore_jurisdiction"].mean()
                    ),
                    1,
                )
                if len(extracted)
                else 0.0
            ),
        }
        tracker.record("groupby", len(extracted), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
