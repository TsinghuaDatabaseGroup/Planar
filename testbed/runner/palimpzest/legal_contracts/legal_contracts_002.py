#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-002."""

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

TASK_ID = "legal_contracts-002"
DATASET = "contract-exhibit"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        semantic_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a Sarbanes-Oxley "
                "certification under Section 302 or Section 906."
            ),
            depends_on=["text"],
        )
        started = time.time()
        semantic_result = semantic_plan.run(config)
        certifications = result_frame(semantic_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            certifications,
            semantic_result,
            time.time() - started,
        )

        answer = int(len(certifications))
        tracker.record("groupby", len(certifications), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
