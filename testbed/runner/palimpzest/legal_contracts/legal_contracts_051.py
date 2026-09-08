#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-051."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_mixed_documents,
    memory_dataset,
    normalize_text_value,
    parse_record_list,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-051"
DATASET = "contract-exhibit"
TARGET_JURISDICTIONS = {
    "cayman islands",
    "british virgin islands",
    "mauritius",
    "luxembourg",
}


def normalize_text(value) -> str | None:
    return normalize_text_value(
        value, null_markers={"", "none", "null", "n/a", "unknown", "unstated"}
    )


def normalize_jurisdiction(value) -> str | None:
    jurisdiction = normalize_text(value)
    if jurisdiction is None:
        return None
    normalized = jurisdiction.casefold().strip()
    if normalized.startswith("the "):
        normalized = normalized[4:]
    canonical = {
        "bvi": "British Virgin Islands",
        "british virgin islands": "British Virgin Islands",
        "cayman islands": "Cayman Islands",
        "mauritius": "Mauritius",
        "luxembourg": "Luxembourg",
    }
    return canonical.get(normalized, jurisdiction)


def normalize_subsidiaries(value) -> list[dict]:
    subsidiaries = []
    for record in parse_record_list(value):
        subsidiary = dict(record)
        subsidiary["jurisdiction"] = normalize_jurisdiction(
            subsidiary.get("jurisdiction")
        )
        subsidiaries.append(subsidiary)
    return subsidiaries


def contains_target(subsidiaries: list[dict]) -> bool:
    return any(
        str(subsidiary.get("jurisdiction", "")).casefold()
        in TARGET_JURISDICTIONS
        for subsidiary in subsidiaries
    )


def project_row(row) -> dict:
    jurisdictions = [
        subsidiary.get("jurisdiction")
        for subsidiary in row["subsidiaries"]
        if subsidiary.get("jurisdiction") is not None
    ]
    counts = Counter(jurisdictions)
    top_three = sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    )[:3]
    return {
        "document_id": row["document_id"],
        "parent_company": row["parent_company"],
        "subsidiary_count": len(row["subsidiaries"]),
        "distinct_jurisdiction_count": len(counts),
        "top_jurisdictions": [
            {"jurisdiction": jurisdiction, "subsidiary_count": count}
            for jurisdiction, count in top_three
        ],
    }


def main() -> None:
    config = get_config(max_tokens=16384)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        filing_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter="Keep the document only if it is a genuine subsidiary-list filing.",
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
                    "name": "parent_company",
                    "type": str | None,
                    "desc": (
                        "The parent company identified by the filing, or null "
                        "when none is identified."
                    ),
                },
                {
                    "name": "subsidiaries",
                    "type": list[dict],
                    "desc": (
                        "One object for every listed subsidiary. Each object has "
                        "the subsidiary name and its stated incorporation "
                        "jurisdiction, using null when the jurisdiction is unstated."
                    ),
                },
            ],
            desc="Extract the identified parent and every listed subsidiary.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result, filings, ["parent_company", "subsidiaries"]
        )
        extracted["parent_company"] = extracted["parent_company"].map(
            normalize_text
        )
        extracted["subsidiaries"] = extracted["subsidiaries"].map(
            normalize_subsidiaries
        )
        tracker.record_semantic(
            "sem_map",
            len(filings),
            extracted,
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["parent_company"].notna()
            & extracted["subsidiaries"].map(contains_target)
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), qualifying)

        projected = [project_row(row) for row in df_records(qualifying)]
        tracker.record("project", len(qualifying), projected)
        answer = projected

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
