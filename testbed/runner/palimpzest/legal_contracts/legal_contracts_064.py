#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-064."""

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
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-064"
DATASET = "contract-exhibit"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        exception_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure "
                "agreement, unilateral non-disclosure agreement, or "
                "confidentiality-and-standstill agreement and contains all six "
                "operative confidentiality carve-outs for public information, "
                "prior knowledge, independent development, lawful unrestricted "
                "third-party receipt, legal compulsion, and disclosure with consent."
            ),
            depends_on=["text"],
        )
        started = time.time()
        exception_result = exception_plan.run(config)
        complete_exceptions = result_frame(exception_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), complete_exceptions, exception_result,
            time.time() - started,
        )

        profile_plan = memory_dataset(
            f"{TASK_ID}-profile", complete_exceptions
        ).sem_filter(
            filter=(
                "Keep the agreement only if it also contains injunctive relief, "
                "an intellectual-property ownership clause, and a clause stating "
                "that confidentiality obligations survive termination."
            ),
            depends_on=["text"],
        )
        started = time.time()
        profile_result = profile_plan.run(config)
        complete_profiles = result_frame(profile_result, complete_exceptions)
        tracker.record_semantic(
            "sem_filter", len(complete_exceptions), complete_profiles,
            profile_result, time.time() - started,
        )

        projected = complete_profiles[
            ["document_id", "word_count"]
        ].reset_index(drop=True)
        tracker.record("project", len(complete_profiles), projected)

        ordered = projected.sort_values(
            ["word_count", "document_id"],
            ascending=[True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(projected), ordered)

        limited = ordered.head(15).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)
        answer = df_records(limited)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
