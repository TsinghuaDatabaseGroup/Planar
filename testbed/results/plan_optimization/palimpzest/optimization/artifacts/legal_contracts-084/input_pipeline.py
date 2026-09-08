#!/usr/bin/env python3
"""Plan-optimization pipeline for legal_contracts-084."""

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
    run_plan_optimization,
    save_output,
)

TASK_ID = "legal_contracts-084"


def at_least_seventy_subsidiaries(record: dict) -> bool:
    try:
        return int(record.get("subsidiary_count")) >= 70
    except (TypeError, ValueError):
        return False


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
    resume_intermediate = os.getenv(
        "PZ_RESUME_OPERATION_INTERMEDIATE",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    raw_output_columns = (
        "sox_company",
        "set(certifying_officer)",
        "covered_person_category",
        "max(subsidiary_count)",
    )
    restored = (
        restore_operation_intermediate(
            TASK_ID,
            "optimized_plan",
            output_columns=raw_output_columns,
        )
        if resume_intermediate
        else None
    )
    reported_elapsed = None

    with Timer() as timer:
        documents = load_mixed_documents("contract-exhibit")

        sox = memory_dataset(f"{TASK_ID}-sox", documents).sem_filter(
            "Keep this document only if it is a SOX Section 302 certification.",
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
            "Keep this document only if it is a clawback or compensation-recovery policy.",
            depends_on=["text"],
        )
        clawback = clawback.sem_filter(
            "Keep this policy only if it permits recovery for misconduct.",
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

        subsidiaries = memory_dataset(
            f"{TASK_ID}-subsidiaries", documents
        ).sem_filter(
            "Keep this document only if it is a subsidiary-list filing.",
            depends_on=["text"],
        )
        subsidiaries = subsidiaries.sem_map(
            cols=[
                {
                    "name": "subsidiary_company",
                    "type": str,
                    "desc": "The parent company entity named by the filing.",
                },
                {
                    "name": "subsidiary_count",
                    "type": int,
                    "desc": "The number of subsidiaries explicitly listed.",
                },
            ],
            desc="Extract the parent company and number of listed subsidiaries.",
            depends_on=["text"],
        )
        subsidiaries = subsidiaries.filter(
            at_least_seventy_subsidiaries,
            depends_on=["subsidiary_count"],
        )
        subsidiaries = subsidiaries.groupby(
            pz.GroupBySig(
                group_by_fields=["subsidiary_company"],
                agg_funcs=["max"],
                agg_fields=["subsidiary_count"],
            )
        )

        company_policies = sox.sem_join(
            clawback,
            condition=(
                "Match records only when sox_company and clawback_company "
                "denote the same company entity despite naming, punctuation, "
                "or corporate-suffix variation."
            ),
            depends_on=["sox_company", "clawback_company"],
        )
        plan = company_policies.sem_join(
            subsidiaries,
            condition=(
                "Match the policy company to subsidiary_company only when they "
                "denote the same company entity despite naming, punctuation, "
                "or corporate-suffix variation."
            ),
            depends_on=["sox_company", "subsidiary_company"],
        )

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
            started = time.time()
            optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
            if optimized.result is None:
                return
            output = optimized.result.to_df().reset_index(drop=True).reindex(
                columns=raw_output_columns
            )
            tracker.record_semantic(
                "optimized_plan",
                {
                    "sox": len(documents),
                    "clawback": len(documents),
                    "subsidiaries": len(documents),
                },
                output,
                optimized.result,
                time.time() - started,
            )
        output = output.rename(
            columns={
                "sox_company": "company",
                "set(certifying_officer)": "certifying_officers",
                "max(subsidiary_count)": "largest_subsidiary_count",
            }
        )
        output = output.reindex(
            columns=[
                "company",
                "certifying_officers",
                "covered_person_category",
                "largest_subsidiary_count",
            ]
        )
        output["certifying_officers"] = output["certifying_officers"].map(
            normalize_officers
        )
        ordered = output.sort_values(
            ["largest_subsidiary_count", "company"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("order_by", len(output), ordered)
        answer = df_records(
            ordered[
                [
                    "company",
                    "certifying_officers",
                    "covered_person_category",
                    "largest_subsidiary_count",
                ]
            ]
        )

    save_output(
        TASK_ID,
        answer,
        reported_elapsed if reported_elapsed is not None else timer.elapsed,
        tracker,
    )


if __name__ == "__main__":
    main()
