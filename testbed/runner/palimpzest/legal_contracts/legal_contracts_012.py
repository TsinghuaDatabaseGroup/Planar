#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-012."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_mixed_documents,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-012"
DATASET = "contract-exhibit"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        clause_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a genuine Sarbanes-Oxley "
                "Section 302 certification and contains the material-changes "
                "disclosure clause for internal control over financial reporting."
            ),
            depends_on=["text"],
        )
        started = time.time()
        clause_result = clause_plan.run(config)
        certifications = result_frame(clause_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            certifications,
            clause_result,
            time.time() - started,
        )

        generic_plan = memory_dataset(
            f"{TASK_ID}-generic", certifications
        ).sem_filter(
            filter=(
                "Keep the certification only if its material-changes clause "
                "uses only generic certification language and describes no "
                "specific change to internal control over financial reporting."
            ),
            depends_on=["text"],
        )
        started = time.time()
        generic_result = generic_plan.run(config)
        generic_only = result_frame(generic_result, certifications)
        tracker.record_semantic(
            "sem_filter",
            len(certifications),
            generic_only,
            generic_result,
            time.time() - started,
        )

        answer = int(len(generic_only))
        tracker.record("groupby", len(generic_only), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
