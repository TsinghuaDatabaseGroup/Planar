#!/usr/bin/env python3
"""Plan-optimization pipeline for legal_contracts-088."""

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
    run_plan_optimization,
    save_output,
)

TASK_ID = "legal_contracts-088"


def normalize_officers(value) -> list[str]:
    values = value if isinstance(value, (set, list, tuple)) else [value]
    return sorted(
        {
            str(item).strip()
            for item in values
            if item is not None and str(item).strip()
        }
    )


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        documents = load_mixed_documents("contract-exhibit")

        sox = memory_dataset(f"{TASK_ID}-sox", documents).sem_filter(
            "Keep this document only if it is an EX-31 SOX officer certification.",
            depends_on=["text"],
        )
        sox = sox.sem_map(
            cols=[
                {
                    "name": "sox_company",
                    "type": str,
                    "desc": "The company entity certified by the document.",
                },
                {
                    "name": "certifying_officer",
                    "type": str,
                    "desc": "The certifying officer name together with title.",
                },
            ],
            desc="Extract the certified company and certifying officer with title.",
            depends_on=["text"],
        )
        sox = sox.groupby(
            pz.GroupBySig(
                group_by_fields=["sox_company"],
                agg_funcs=["set"],
                agg_fields=["certifying_officer"],
            )
        )

        clawback = memory_dataset(f"{TASK_ID}-clawback", documents).sem_filter(
            "Keep this document only if it is an EX-97 clawback policy.",
            depends_on=["text"],
        )
        clawback = clawback.sem_filter(
            (
                "Keep this policy only if it provides financial-restatement "
                "recovery and also permits recovery for misconduct."
            ),
            depends_on=["text"],
        )
        clawback = clawback.sem_map(
            cols=[
                {
                    "name": "clawback_company",
                    "type": str,
                    "desc": "The company entity covered by the policy.",
                },
                {
                    "name": "covered_person_category",
                    "type": str,
                    "desc": "The normalized covered-person category.",
                },
            ],
            desc="Extract the company and normalize its covered-person category.",
            depends_on=["text"],
        )
        clawback = clawback.distinct(
            ["clawback_company", "covered_person_category"]
        )

        plan = sox.sem_join(
            clawback,
            condition=(
                "Match records only when sox_company and clawback_company "
                "denote the same company entity despite naming, punctuation, "
                "or corporate-suffix variation."
            ),
            depends_on=["sox_company", "clawback_company"],
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True)
        tracker.record_semantic(
            "optimized_plan",
            {"sox": len(documents), "clawback": len(documents)},
            output,
            optimized.result,
            time.time() - started,
        )
        output = output.rename(
            columns={
                "sox_company": "company",
                "set(certifying_officer)": "certifying_officers",
            }
        )
        output["certifying_officers"] = output["certifying_officers"].map(
            normalize_officers
        )
        answer = df_records(
            output[["company", "certifying_officers", "covered_person_category"]]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
