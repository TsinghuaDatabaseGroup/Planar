#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-040."""

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

TASK_ID = "legal_contracts-040"
DATASET = "contract-exhibit"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        mutual_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure "
                "agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        mutual_result = mutual_plan.run(config)
        mutual_agreements = result_frame(mutual_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            mutual_agreements,
            mutual_result,
            time.time() - started,
        )

        software_plan = memory_dataset(
            f"{TASK_ID}-software", mutual_agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if its definition of confidential "
                "information explicitly includes source code or software."
            ),
            depends_on=["text"],
        )
        started = time.time()
        software_result = software_plan.run(config)
        software_agreements = result_frame(
            software_result, mutual_agreements
        )
        tracker.record_semantic(
            "sem_filter",
            len(mutual_agreements),
            software_agreements,
            software_result,
            time.time() - started,
        )

        retention_plan = memory_dataset(
            f"{TASK_ID}-retention", software_agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it permits the recipient to retain "
                "at least one archival copy of materials after termination."
            ),
            depends_on=["text"],
        )
        started = time.time()
        retention_result = retention_plan.run(config)
        qualifying = result_frame(retention_result, software_agreements)
        tracker.record_semantic(
            "sem_filter",
            len(software_agreements),
            qualifying,
            retention_result,
            time.time() - started,
        )

        answer = "Yes" if len(qualifying) > 0 else "No"
        tracker.record("groupby", len(qualifying), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
