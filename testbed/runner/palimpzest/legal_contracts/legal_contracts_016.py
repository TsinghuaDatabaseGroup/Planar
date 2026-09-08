#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for legal_contracts-016."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    DATA_ROOT,
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_html_documents,
    memory_dataset,
    result_frame,
    run_optimized_plan,
    save_output,
)

TASK_ID = "legal_contracts-016"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"


def main() -> None:
    config = get_config(
        max_tokens=4096,
        all_optimizations=True,
        include_small_model=True,
    )
    tracker = StepTracker(TASK_ID, optimizer_strategy=OPTIMIZER)

    with Timer() as timer:
        document_root = DATA_ROOT / "contract-exhibit"
        documents = pd.DataFrame.from_records(
            [
                {
                    "contract_id": path.name,
                    "doc_format": path.suffix.lower().lstrip("."),
                }
                for path in sorted(document_root.iterdir())
                if path.is_file()
            ]
        )
        tracker.record("scan", None, documents)

        html_documents = documents.loc[documents["doc_format"] == "htm"].reset_index(drop=True)
        tracker.record("filter", len(documents), html_documents)

        reports = load_html_documents(
            "contract-exhibit",
            filenames=html_documents["contract_id"],
        )
        tracker.record("scan", len(html_documents), reports)

        plan = memory_dataset(TASK_ID, reports).sem_filter(
            filter=(
                "Keep documents that substantively function as a nondisclosure "
                "or confidentiality document, including documents that also "
                "serve another SEC exhibit purpose. Exclude documents that only "
                "mention confidentiality without imposing a substantive "
                "confidentiality regime."
            ),
            depends_on=["filename", "text"],
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        matched = result_frame(semantic_result)
        memberships = matched[["filename"]].rename(columns={"filename": "contract_id"})
        memberships["doc_format"] = "htm"
        tracker.record_semantic(
            "sem_filter",
            len(html_documents),
            memberships,
            semantic_result,
            time.time() - started,
        )

        grouped = pd.DataFrame.from_records([{"document_count": int(memberships["contract_id"].nunique())}])
        tracker.record("groupby", len(memberships), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
