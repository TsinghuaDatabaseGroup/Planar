#!/usr/bin/env python3
"""Palimpzest pipeline for finance-010."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "finance-010"
DATASET = "SEC"


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(DATASET, "CSV/2023.csv", ["text"])
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        extraction_plan = memory_dataset(
            f"{TASK_ID}-mentions", documents
        ).sem_map(
            cols=[
                {
                    "name": "esg_mentioned",
                    "type": bool,
                    "desc": (
                        "True when the filing mentions environmental, social, or "
                        "governance topics or sustainability; false otherwise."
                    ),
                },
                {
                    "name": "ai_ml_mentioned",
                    "type": bool,
                    "desc": (
                        "True when the filing contains a qualifying AI/ML technology "
                        "reference in the context of the company's business, "
                        "products, or strategy; false otherwise."
                    ),
                },
            ],
            desc=(
                "Determine whether the filing mentions ESG or sustainability topics "
                "and whether it contains a qualifying AI/ML reference."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            documents,
            ["esg_mentioned", "ai_ml_mentioned"],
        )
        extracted["esg_mentioned"] = extracted["esg_mentioned"].map(parse_bool)
        extracted["ai_ml_mentioned"] = extracted["ai_ml_mentioned"].map(
            parse_bool
        )
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        esg = extracted["esg_mentioned"]
        ai_ml = extracted["ai_ml_mentioned"]
        answer = {
            "both_esg_and_ai_ml": int((esg & ai_ml).sum()),
            "esg_only": int((esg & ~ai_ml).sum()),
            "ai_ml_only": int((~esg & ai_ml).sum()),
            "neither": int((~esg & ~ai_ml).sum()),
        }
        tracker.record("groupby", len(extracted), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
