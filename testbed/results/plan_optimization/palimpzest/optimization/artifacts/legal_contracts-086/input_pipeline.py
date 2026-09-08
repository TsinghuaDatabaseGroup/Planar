#!/usr/bin/env python3
"""Plan-optimization pipeline for legal_contracts-086."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import palimpzest as pz  # noqa: E402
from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_mixed_documents,
    memory_dataset,
    restore_operation_intermediate,
    result_frame,
    run_plan_optimization,
    save_output,
)

TASK_ID = "legal_contracts-086"


def has_governing_law(record: dict) -> bool:
    value = record.get("governing_law")
    return value is not None and bool(str(value).strip())


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    resume_intermediate = os.getenv(
        "PZ_RESUME_OPERATION_INTERMEDIATE",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    restored = (
        restore_operation_intermediate(
            TASK_ID,
            "optimized_plan",
            output_columns=("governing_law", "matching_count"),
        )
        if resume_intermediate
        else None
    )
    reported_elapsed = None

    with Timer() as timer:
        if restored is not None:
            operation, output, source_path = restored
            tracker.restore_operation(operation, output, source_path)
            reported_elapsed = float(operation.get("elapsed_seconds", 0.0))
            print(
                f"[RESUME] {TASK_ID} optimized_plan restored from "
                f"{source_path}",
                flush=True,
            )
        else:
            config = get_config(max_tokens=4096)
            documents = load_mixed_documents("contract-exhibit")
            plan = memory_dataset(TASK_ID, documents).sem_filter(
                (
                    "Keep this document only if it is an NDA, including a dual-"
                    "labelled NDA exhibit, with an effective date strictly after "
                    "2010-01-01."
                ),
                depends_on=["text"],
            )
            plan = plan.sem_filter(
                (
                    "Keep this NDA only if its confidential-information definition "
                    "explicitly includes electronic data or records, email, "
                    "software or code, metadata, or other computer-readable "
                    "material."
                ),
                depends_on=["text"],
            )
            plan = plan.sem_filter(
                (
                    "Keep this NDA only if it requires destruction of at least some "
                    "confidential materials or copies. Exclude clauses requiring "
                    "only return or merely allowing an unrestricted choice between "
                    "return and destruction."
                ),
                depends_on=["text"],
            )
            plan = plan.sem_filter(
                (
                    "Keep this NDA only if confidentiality or non-use obligations "
                    "expressly survive termination or expiry or continue beyond "
                    "the stated term."
                ),
                depends_on=["text"],
            )
            plan = plan.sem_map(
                cols=[
                    {
                        "name": "governing_law",
                        "type": str | None,
                        "desc": (
                            "Canonical expressly stated governing-law jurisdiction, "
                            "or null when it is unstated."
                        ),
                    }
                ],
                desc="Extract and canonicalize the governing-law jurisdiction.",
                depends_on=["text"],
            )
            plan = plan.filter(
                has_governing_law,
                depends_on=["governing_law"],
            )
            plan = plan.groupby(
                pz.GroupBySig(
                    group_by_fields=["governing_law"],
                    agg_funcs=["count"],
                    agg_fields=["document_id"],
                )
            )

            started = time.time()
            optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
            if optimized.result is None:
                return
            output = result_frame(
                optimized.result,
                source=documents,
                generated_columns=(
                    "governing_law",
                    "count(document_id)",
                ),
            ).rename(columns={"count(document_id)": "matching_count"})
            output = output.reindex(
                columns=["governing_law", "matching_count"]
            )
            tracker.record_semantic(
                "optimized_plan",
                len(documents),
                output,
                optimized.result,
                time.time() - started,
            )
        ordered = output.sort_values(
            ["matching_count", "governing_law"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("order_by", len(output), ordered)
        answer = df_records(ordered[["governing_law", "matching_count"]])

    save_output(
        TASK_ID,
        answer,
        reported_elapsed if reported_elapsed is not None else timer.elapsed,
        tracker,
    )


if __name__ == "__main__":
    main()
