#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-077."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from collections import defaultdict

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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-077"
DATASET = "contract-exhibit"
LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "llp",
    "lp",
    "ltd",
    "plc",
}


def normalize_number(value) -> float | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def normalize_string_list(value) -> list[str]:
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return []
    parsed = value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value.strip())
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
    if pd.api.types.is_list_like(parsed) and not isinstance(parsed, dict):
        parsed = list(parsed)
    else:
        parsed = [parsed]
    return [
        normalized
        for item in parsed
        if (normalized := " ".join(str(item).strip().split()))
    ]


def normalize_company_name(value) -> str:
    tokens = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def build_party_index(records: list[dict], party_column: str):
    index = defaultdict(list)
    for record in records:
        seen = set()
        for stated_name in record[party_column]:
            key = normalize_company_name(stated_name)
            identity = (key, stated_name)
            if not key or identity in seen:
                continue
            seen.add(identity)
            index[key].append((record, stated_name))
    return index


def join_records(nda_records: list[dict], restrictive_records: list[dict]) -> list[dict]:
    restrictive_index = build_party_index(
        restrictive_records, "restrictive_company_parties"
    )
    matches = {}
    for nda in nda_records:
        for nda_name in nda["nda_company_parties"]:
            key = normalize_company_name(nda_name)
            if not key:
                continue
            for restrictive, restrictive_name in restrictive_index.get(key, []):
                if nda["nda_document_id"] == restrictive["restrictive_document_id"]:
                    continue
                pair_key = (
                    nda["nda_document_id"],
                    restrictive["restrictive_document_id"],
                )
                match = matches.setdefault(
                    pair_key,
                    {
                        "matched_stated_variants": set(),
                        "nda_document_id": nda["nda_document_id"],
                        "restrictive_document_id": restrictive[
                            "restrictive_document_id"
                        ],
                        "nda_duration_years": nda["nda_duration_years"],
                        "noncompete_duration_years": restrictive[
                            "noncompete_duration_years"
                        ],
                    },
                )
                match["matched_stated_variants"].update(
                    (nda_name, restrictive_name)
                )
    output = []
    for match in matches.values():
        normalized = dict(match)
        normalized["matched_stated_variants"] = sorted(
            normalized["matched_stated_variants"]
        )
        output.append(normalized)
    return output


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        nda_docs = load_mixed_documents(DATASET)
        tracker.record("scan", None, nda_docs)

        nda_filter_plan = memory_dataset(
            f"{TASK_ID}-nda-filter", nda_docs
        ).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure "
                "agreement, unilateral non-disclosure agreement, or "
                "confidentiality-and-standstill agreement with a finite stated "
                "confidentiality term."
            ),
            depends_on=["text"],
        )
        started = time.time()
        nda_filter_result = nda_filter_plan.run(config)
        nda_agreements = result_frame(nda_filter_result, nda_docs)
        tracker.record_semantic(
            "sem_filter", len(nda_docs), nda_agreements, nda_filter_result,
            time.time() - started,
        )

        nda_extraction_plan = memory_dataset(
            f"{TASK_ID}-nda-extraction", nda_agreements
        ).sem_map(
            cols=[
                {
                    "name": "nda_company_parties",
                    "type": list[str],
                    "desc": "All corporate party names stated in the agreement.",
                },
                {
                    "name": "nda_duration_years",
                    "type": float | None,
                    "desc": "The finite confidentiality duration normalized to years.",
                },
            ],
            desc="Extract NDA corporate parties and confidentiality duration.",
            depends_on=["text"],
        )
        started = time.time()
        nda_extraction_result = nda_extraction_plan.run(config)
        nda_extracted = result_frame(
            nda_extraction_result,
            nda_agreements,
            ["nda_company_parties", "nda_duration_years"],
        )
        nda_extracted["nda_company_parties"] = nda_extracted[
            "nda_company_parties"
        ].map(normalize_string_list)
        nda_extracted["nda_duration_years"] = nda_extracted[
            "nda_duration_years"
        ].map(normalize_number)
        nda_extracted = nda_extracted.rename(
            columns={"document_id": "nda_document_id"}
        )[
            ["nda_document_id", "nda_company_parties", "nda_duration_years"]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(nda_agreements), nda_extracted, nda_extraction_result,
            time.time() - started,
        )

        restrictive_docs = load_mixed_documents(DATASET)
        tracker.record("scan", None, restrictive_docs)

        restrictive_filter_plan = memory_dataset(
            f"{TASK_ID}-restrictive-filter", restrictive_docs
        ).sem_filter(
            filter=(
                "Keep the document only if it is an employment agreement or a "
                "standalone restrictive-covenant agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        restrictive_filter_result = restrictive_filter_plan.run(config)
        restrictive_agreements = result_frame(
            restrictive_filter_result, restrictive_docs
        )
        tracker.record_semantic(
            "sem_filter", len(restrictive_docs), restrictive_agreements,
            restrictive_filter_result, time.time() - started,
        )

        noncompete_plan = memory_dataset(
            f"{TASK_ID}-noncompete", restrictive_agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains an operative non-compete "
                "clause with a finite stated duration."
            ),
            depends_on=["text"],
        )
        started = time.time()
        noncompete_result = noncompete_plan.run(config)
        finite_noncompetes = result_frame(
            noncompete_result, restrictive_agreements
        )
        tracker.record_semantic(
            "sem_filter", len(restrictive_agreements), finite_noncompetes,
            noncompete_result, time.time() - started,
        )

        restrictive_extraction_plan = memory_dataset(
            f"{TASK_ID}-restrictive-extraction", finite_noncompetes
        ).sem_map(
            cols=[
                {
                    "name": "restrictive_company_parties",
                    "type": list[str],
                    "desc": "All corporate party names stated in the agreement.",
                },
                {
                    "name": "noncompete_duration_years",
                    "type": float | None,
                    "desc": "The finite non-compete duration normalized to years.",
                },
            ],
            desc=(
                "Extract restrictive-agreement corporate parties and the "
                "non-compete duration."
            ),
            depends_on=["text"],
        )
        started = time.time()
        restrictive_extraction_result = restrictive_extraction_plan.run(config)
        restrictive_extracted = result_frame(
            restrictive_extraction_result,
            finite_noncompetes,
            ["restrictive_company_parties", "noncompete_duration_years"],
        )
        restrictive_extracted["restrictive_company_parties"] = (
            restrictive_extracted["restrictive_company_parties"].map(
                normalize_string_list
            )
        )
        restrictive_extracted["noncompete_duration_years"] = (
            restrictive_extracted["noncompete_duration_years"].map(
                normalize_number
            )
        )
        restrictive_extracted = restrictive_extracted.rename(
            columns={"document_id": "restrictive_document_id"}
        )[
            [
                "restrictive_document_id",
                "restrictive_company_parties",
                "noncompete_duration_years",
            ]
        ].reset_index(drop=True)
        tracker.record_semantic(
            "sem_map", len(finite_noncompetes), restrictive_extracted,
            restrictive_extraction_result, time.time() - started,
        )

        nda_records = df_records(nda_extracted)
        restrictive_records = df_records(restrictive_extracted)
        joined = join_records(nda_records, restrictive_records)
        tracker.record(
            "join",
            {
                "nda_records": len(nda_records),
                "restrictive_records": len(restrictive_records),
            },
            joined,
        )

        projected = [
            {
                "company_name": record["matched_stated_variants"][0],
                "nda_document_id": record["nda_document_id"],
                "restrictive_document_id": record["restrictive_document_id"],
                "nda_duration_years": record["nda_duration_years"],
                "noncompete_duration_years": record[
                    "noncompete_duration_years"
                ],
            }
            for record in joined
        ]
        tracker.record("project", len(joined), projected)
        answer = projected

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
