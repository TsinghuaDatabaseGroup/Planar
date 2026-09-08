#!/usr/bin/env python3
"""Plan-optimization pipeline for legal_contracts-060."""

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
    run_plan_optimization,
    save_output,
)

TASK_ID = "legal_contracts-060"


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        documents = load_mixed_documents("contract-exhibit")
        plan = memory_dataset(TASK_ID, documents).sem_filter(
            (
                "Keep this document only if it is a non-disclosure or "
                "confidentiality agreement."
            ),
            depends_on=["text"],
        )
        plan = plan.sem_filter(
            (
                "Keep this agreement only if it contains a residual-"
                "information clause permitting use of information retained in "
                "unaided memory and expressly states a governing law."
            ),
            depends_on=["text"],
        )
        plan = plan.sem_map(
            cols=[
                {
                    "name": "non_disclosure_direction",
                    "type": str,
                    "desc": (
                        "The operative nondisclosure direction, exactly mutual "
                        "or unilateral."
                    ),
                },
                {
                    "name": "governing_law",
                    "type": str,
                    "desc": "The expressly stated governing-law jurisdiction.",
                },
            ],
            desc=(
                "Extract the agreement's nondisclosure direction and expressly "
                "stated governing law."
            ),
            depends_on=["document_id", "text"],
        )
        plan = plan.project(
            ["document_id", "non_disclosure_direction", "governing_law"]
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True)
        tracker.record_semantic(
            "optimized_plan",
            len(documents),
            output,
            optimized.result,
            time.time() - started,
        )
        answer = df_records(
            output[
                [
                    "document_id",
                    "non_disclosure_direction",
                    "governing_law",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
