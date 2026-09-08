#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-052."""

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

TASK_ID = "legal_contracts-052"
DATASET = "contract-exhibit"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure "
                "agreement, unilateral non-disclosure agreement, or "
                "confidentiality-and-standstill agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        agreement_result = agreement_plan.run(config)
        agreements = result_frame(agreement_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), agreements, agreement_result,
            time.time() - started,
        )

        assignment_plan = memory_dataset(
            f"{TASK_ID}-assignment", agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if an assignment provision permits "
                "assignment of the agreement or its contractual rights or "
                "obligations to a corporate affiliate without the counterparty's "
                "consent."
            ),
            depends_on=["text"],
        )
        started = time.time()
        assignment_result = assignment_plan.run(config)
        assignable = result_frame(assignment_result, agreements)
        tracker.record_semantic(
            "sem_filter", len(agreements), assignable, assignment_result,
            time.time() - started,
        )

        forum_plan = memory_dataset(
            f"{TASK_ID}-forum", assignable
        ).sem_filter(
            filter=(
                "Keep the agreement only if a forum-selection, venue, "
                "jurisdiction, or enforcement provision expressly identifies a "
                "United States federal court as a forum for disputes under the "
                "agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        forum_result = forum_plan.run(config)
        federal_forum = result_frame(forum_result, assignable)
        tracker.record_semantic(
            "sem_filter", len(assignable), federal_forum, forum_result,
            time.time() - started,
        )

        answer = int(len(federal_forum))
        tracker.record("groupby", len(federal_forum), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
