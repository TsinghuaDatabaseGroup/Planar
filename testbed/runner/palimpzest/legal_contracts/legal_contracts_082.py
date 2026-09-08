#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-082."""

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

TASK_ID = "legal_contracts-082"
DATASET = "contract-exhibit"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure agreement, "
                "a unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        perpetual_plan = memory_dataset(
            f"{TASK_ID}-perpetual", scoped
        ).sem_filter(
            filter=(
                "Keep the agreement only if its confidentiality obligation is "
                "expressly perpetual or indefinite."
            ),
            depends_on=["text"],
        )
        started = time.time()
        perpetual_result = perpetual_plan.run(config)
        perpetual = result_frame(perpetual_result, scoped)
        tracker.record_semantic(
            "sem_filter", len(scoped), perpetual, perpetual_result,
            time.time() - started,
        )

        exception_plan = memory_dataset(
            f"{TASK_ID}-no-independent-development", perpetual
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains no operative "
                "confidentiality exception for independently developed information."
            ),
            depends_on=["text"],
        )
        started = time.time()
        exception_result = exception_plan.run(config)
        no_exception = result_frame(exception_result, perpetual)
        tracker.record_semantic(
            "sem_filter", len(perpetual), no_exception, exception_result,
            time.time() - started,
        )

        severability_plan = memory_dataset(
            f"{TASK_ID}-no-severability", no_exception
        ).sem_filter(
            filter="Keep the agreement only if it contains no severability clause.",
            depends_on=["text"],
        )
        started = time.time()
        severability_result = severability_plan.run(config)
        no_severability = result_frame(severability_result, no_exception)
        tracker.record_semantic(
            "sem_filter", len(no_exception), no_severability, severability_result,
            time.time() - started,
        )

        projected = no_severability[["document_id"]].reset_index(drop=True)
        tracker.record("project", len(no_severability), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
