#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-056."""

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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-056"
DATASET = "contract-exhibit"
AGREEMENT_TYPES = (
    "mutual_nda",
    "unilateral_nda",
    "confidentiality_standstill",
    "not_in_scope",
)


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

        labeling_plan = memory_dataset(
            f"{TASK_ID}-classification", agreements
        ).sem_map(
            cols=[
                {
                    "name": "agreement_type",
                    "type": str | None,
                    "desc": (
                        "Exactly one of mutual_nda, unilateral_nda, "
                        "confidentiality_standstill, or not_in_scope, based on "
                        "the direction of confidentiality protection and whether "
                        "the agreement includes a transaction standstill."
                    ),
                }
            ],
            desc="Assign each document to exactly one requested agreement-type label.",
            depends_on=["text"],
        )
        started = time.time()
        labeling_result = labeling_plan.run(config)
        labeled = result_frame(labeling_result, agreements, ["agreement_type"])
        labeled["agreement_type"] = labeled["agreement_type"].map(
            lambda value: normalize_enum(value, AGREEMENT_TYPES)
        )
        tracker.record_semantic(
            "sem_map", len(agreements), labeled, labeling_result,
            time.time() - started,
        )

        projected = labeled[["agreement_type", "word_count"]].rename(
            columns={"word_count": "document_word_count"}
        ).reset_index(drop=True)
        tracker.record("project", len(labeled), projected)

        grouped = (
            projected.groupby("agreement_type", sort=False, dropna=False)
            .agg(
                document_count=("agreement_type", "size"),
                avg_word_count=("document_word_count", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_word_count"] = grouped["avg_word_count"].round(2)
        tracker.record("groupby", len(projected), grouped)

        ordered = grouped.sort_values(
            ["document_count", "agreement_type"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)

        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
