#!/usr/bin/env python3
"""Plan-optimization pipeline for legal_contracts-012."""

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
    run_plan_optimization,
    save_output,
)

TASK_ID = "legal_contracts-012"


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        documents = load_mixed_documents("contract-exhibit")
        plan = memory_dataset(TASK_ID, documents).sem_filter(
            (
                "Keep this document only if it is a genuine SOX Section 302 "
                "certification containing the material-changes disclosure "
                "clause for internal control over financial reporting."
            ),
            depends_on=["text"],
        )
        plan = plan.sem_filter(
            (
                "Keep this certification only if that clause uses generic "
                "certification language and describes no specific control "
                "change."
            ),
            depends_on=["text"],
        )
        plan = plan.count()

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
        answer = int(output.iloc[0]["count"]) if not output.empty else 0

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
