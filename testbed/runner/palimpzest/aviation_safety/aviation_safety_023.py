#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for aviation_safety-023."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    result_frame,
    run_optimized_plan,
    save_output,
)

TASK_ID = "aviation_safety-023"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
FACTORS = (
    ("Communication Breakdown", "communication_breakdown"),
    ("Workload", "workload"),
    ("Confusion", "confusion"),
    ("Situational Awareness", "situational_awareness"),
)
JOIN_COLUMNS = [
    "incident_id",
    "text",
    "factor_incident_id",
    "canonical_human_factor",
]


def factor_branch(
    needle: str,
    canonical_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors = load_table(
        "asrs",
        "person_factors.csv",
        ["incident_id", "factor_type", "value"],
    )
    selected = factors.loc[
        (factors["factor_type"] == "human_factors") & factors["value"].str.contains(needle, regex=False, na=False)
    ].reset_index(drop=True)
    projected = selected[["incident_id"]].copy()
    projected["canonical_human_factor"] = canonical_label
    return factors, selected, projected


def main() -> None:
    config = get_config(
        max_tokens=4096,
        all_optimizations=True,
        include_small_model=True,
    )
    tracker = StepTracker(TASK_ID, optimizer_strategy=OPTIMIZER)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem"],
        )
        tracker.record("scan", None, incidents)
        human_factors = incidents.loc[incidents["primary_problem"] == "Human Factors"].reset_index(drop=True)
        tracker.record("filter", len(incidents), human_factors)
        incident_docs = load_selected_texts("asrs", human_factors)[["incident_id", "text"]]
        tracker.record("scan", len(human_factors), incident_docs)

        branches = []
        for needle, canonical_label in FACTORS:
            source, selected, branch = factor_branch(needle, canonical_label)
            tracker.record("scan", None, source)
            tracker.record("filter", len(source), selected)
            tracker.record("project", len(selected), branch)
            branches.append(branch)
        factor_rows = pd.concat(branches, ignore_index=True)
        tracker.record(
            "union",
            {f"branch_{index + 1}": len(branch) for index, branch in enumerate(branches)},
            factor_rows,
        )
        factor_rows = factor_rows.drop_duplicates(["incident_id", "canonical_human_factor"]).reset_index(drop=True)
        tracker.record("dedup", sum(map(len, branches)), factor_rows)
        factor_rows = factor_rows.rename(columns={"incident_id": "factor_incident_id"})

        plan = memory_dataset(
            f"{TASK_ID}-incidents",
            incident_docs,
        ).sem_join(
            memory_dataset(f"{TASK_ID}-factors", factor_rows),
            condition=(
                "Evaluate only rows where incident_id equals factor_incident_id. "
                "Keep an incident-factor pair only when the narrative portrays "
                "canonical_human_factor as causally contributing to the event, "
                "rather than merely mentioning it or providing background context."
            ),
            depends_on=JOIN_COLUMNS,
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        matched = result_frame(semantic_result)
        if matched.empty:
            matched = pd.DataFrame(columns=JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(incident_docs), "right": len(factor_rows)},
            matched,
            semantic_result,
            time.time() - started,
        )

        distinct_pairs = matched.drop_duplicates(["incident_id", "canonical_human_factor"]).reset_index(drop=True)
        rows = []
        for _, label in FACTORS:
            subset = distinct_pairs.loc[distinct_pairs["canonical_human_factor"] == label]
            if subset.empty:
                continue
            rows.append(
                {
                    "canonical_human_factor": label,
                    "incident_factor_pair_count": len(subset),
                }
            )
        grouped = pd.DataFrame.from_records(
            rows,
            columns=[
                "canonical_human_factor",
                "incident_factor_pair_count",
            ],
        )
        tracker.record("groupby", len(matched), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
