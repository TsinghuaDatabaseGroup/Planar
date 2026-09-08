#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-014."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_mixed_documents,
    memory_dataset,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-014"
DATASET = "contract-exhibit"
CLAUSE_COLUMNS = (
    "has_internal_controls_clause",
    "has_fraud_disclosure_clause",
    "has_material_changes_clause",
)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        certification_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a genuine Sarbanes-Oxley "
                "Section 302 certification."
            ),
            depends_on=["text"],
        )
        started = time.time()
        certification_result = certification_plan.run(config)
        certifications = result_frame(certification_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), certifications, certification_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", certifications
        ).sem_map(
            cols=[
                {
                    "name": "has_internal_controls_clause",
                    "type": bool,
                    "desc": (
                        "True if the certification contains the internal-controls "
                        "responsibility clause; false otherwise."
                    ),
                },
                {
                    "name": "has_fraud_disclosure_clause",
                    "type": bool,
                    "desc": (
                        "True if the certification contains the management-fraud "
                        "disclosure clause; false otherwise."
                    ),
                },
                {
                    "name": "has_material_changes_clause",
                    "type": bool,
                    "desc": (
                        "True if the certification contains the material-changes "
                        "clause; false otherwise."
                    ),
                },
            ],
            desc="Extract the three requested Section 302 clause indicators.",
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result, certifications, CLAUSE_COLUMNS)
        for column in CLAUSE_COLUMNS:
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(certifications), extracted, extraction_result,
            time.time() - started,
        )

        grouped = (
            extracted.groupby(list(CLAUSE_COLUMNS), sort=False, dropna=False)
            .size()
            .rename("certification_count")
            .reset_index()
        )
        tracker.record("groupby", len(extracted), grouped)

        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
