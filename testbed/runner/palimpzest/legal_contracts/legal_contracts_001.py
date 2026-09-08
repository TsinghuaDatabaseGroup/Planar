#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-001."""

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

TASK_ID = "legal_contracts-001"
DATASET = "contract-exhibit"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        short_documents = documents.loc[
            documents["word_count"] < 200
        ].reset_index(drop=True)
        tracker.record("filter", len(documents), short_documents)

        semantic_plan = memory_dataset(TASK_ID, short_documents).sem_filter(
            filter=(
                "Keep the document only if it is an independent auditor "
                "consent letter authorizing the use or incorporation of an "
                "auditor's report in an SEC filing. Exclude legal-counsel, "
                "engineering, petroleum-reserve, and other non-auditor consents."
            ),
            depends_on=["text"],
        )
        started = time.time()
        semantic_result = semantic_plan.run(config)
        auditor_consents = result_frame(semantic_result, short_documents)
        tracker.record_semantic(
            "sem_filter",
            len(short_documents),
            auditor_consents,
            semantic_result,
            time.time() - started,
        )

        answer = int(len(auditor_consents))
        tracker.record("groupby", len(auditor_consents), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
