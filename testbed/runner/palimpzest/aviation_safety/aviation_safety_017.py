#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for aviation_safety-017."""

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

TASK_ID = "aviation_safety-017"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"


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
            [
                "incident_id",
                "text_file",
                "anomaly_summary",
                "result_summary",
            ],
        )
        tracker.record("scan", None, incidents)

        equipment_reports = incidents.loc[
            incidents["anomaly_summary"].str.contains(
                "Aircraft Equipment Problem",
                na=False,
                regex=False,
            )
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), equipment_reports)

        reports = load_selected_texts("asrs", equipment_reports)[["incident_id", "result_summary", "text"]]
        tracker.record("scan", len(equipment_reports), reports)

        plan = memory_dataset(TASK_ID, reports).sem_filter(
            filter=(
                "Keep reports that describe active crew or maintenance "
                "troubleshooting and support a clear resulting status: resolved "
                "when corrective action cleared the problem, intermittent when "
                "it disappeared but remained uncertain or recurred, or still "
                "unsafe when it persisted or left the aircraft unsafe."
            ),
            depends_on=["incident_id", "result_summary", "text"],
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        matched = result_frame(semantic_result, reports)[["incident_id"]].reset_index(drop=True)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            matched,
            semantic_result,
            time.time() - started,
        )

        grouped = pd.DataFrame.from_records([{"count_incidents": int(matched["incident_id"].nunique())}])
        tracker.record("groupby", len(matched), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
