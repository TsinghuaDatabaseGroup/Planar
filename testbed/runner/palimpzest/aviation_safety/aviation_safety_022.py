#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for aviation_safety-022."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_jsonl,
    parse_record_list,
    run_bounded_semantic_aggregate,
    save_output,
)

TASK_ID = "aviation_safety-022"
OPTIMIZER = "pareto (multi-model static estimates; no native aggregate validator)"


def main() -> None:
    config = get_config(
        max_tokens=4096,
        all_optimizations=True,
        include_small_model=True,
    )
    tracker = StepTracker(TASK_ID, optimizer_strategy=OPTIMIZER)

    with Timer() as timer:
        reports = load_jsonl(
            "operator_implement",
            "inputs/NASA_ASRS-030_safety_barrier_candidates.jsonl",
        )[["incident_id", "flight_phase", "aircraft_operator", "text"]]
        tracker.record("scan", None, reports)

        aggregate = run_bounded_semantic_aggregate(
            TASK_ID,
            reports,
            config,
            tracker=tracker,
            col={
                "name": "rows",
                "type": list[dict],
                "desc": (
                    "One object for every represented combination, each with "
                    "exactly the keys \"flight_phase\", "
                    "\"aircraft_operator\", \"candidate_incident_count\" "
                    "(integer), \"successful_barrier_count\" (integer), and "
                    "\"successful_barrier_rate\" (number). Do not include a "
                    "rank field."
                ),
            },
            agg=(
                "Across all input records, count a successful barrier only when "
                "the narrative clearly establishes hazard detection, a concrete "
                "intervention, and avoidance of a worse consequence or "
                "stabilization. Return every flight-phase and aircraft-operator "
                "combination with at least twenty candidates. Include both "
                "counts and the successful-barrier rate for every returned "
                "combination; do not rank or select a top five."
            ),
            depends_on=[
                "incident_id",
                "flight_phase",
                "aircraft_operator",
                "text",
            ],
            partial_col={
                "name": "partial_rows",
                "type": list[dict],
                "desc": (
                    "Additive sufficient-statistic objects, each with exactly "
                    "the keys \"flight_phase\", \"aircraft_operator\", "
                    "\"candidate_incident_count\" (integer), and "
                    "\"successful_barrier_count\" (integer)."
                ),
            },
            partial_agg=(
                "Within only this input chunk, group records by flight_phase "
                "and aircraft_operator. Count distinct candidate incidents and "
                "count a successful barrier only when the narrative clearly "
                "establishes hazard detection, a concrete intervention, and "
                "avoidance of a worse consequence or stabilization. Return one "
                "additive sufficient-statistic object for every represented "
                "combination. Emit each exact combination only once; combine "
                "duplicate group rows before responding. Do not calculate "
                "rates or select a top five yet."
            ),
            partial_depends_on=[
                "flight_phase",
                "aircraft_operator",
                "candidate_incident_count",
                "successful_barrier_count",
            ],
            partial_merge_agg=(
                "These inputs are additive sufficient statistics from disjoint "
                "chunks. Sum candidate_incident_count and "
                "successful_barrier_count for each exact flight_phase and "
                "aircraft_operator combination. Return the same sufficient-"
                "statistic fields for every represented combination. Do not "
                "calculate rates or select a top five yet."
            ),
            merge_agg=(
                "These inputs are additive sufficient statistics from disjoint "
                "chunks of the complete relation. Sum both counts for each exact "
                "flight_phase and aircraft_operator combination and calculate "
                "successful_barrier_rate as successful_barrier_count divided by "
                "candidate_incident_count. Every input combination already has "
                "at least twenty candidates. Return every combination exactly "
                "once; do not rank, select a top five, or omit a combination."
            ),
            partial_max_rows=40,
            merge_max_rows=50,
            merge_sort_by=["flight_phase", "aircraft_operator"],
            final_max_rows=200,
            final_additive_fields=[
                "candidate_incident_count",
                "successful_barrier_count",
            ],
            final_numeric_minimums={"candidate_incident_count": 20},
            final_candidate_max_rows=20,
            final_group_by=["flight_phase", "aircraft_operator"],
        )
        aggregate_frame = aggregate.output
        if aggregate_frame.empty:
            raise ValueError(f"{TASK_ID}: semantic aggregate returned no answer")

        projected_rows = []
        for item in parse_record_list(aggregate_frame.iloc[0]["rows"]):
            candidate_count = int(item.get("candidate_incident_count"))
            success_count = int(item.get("successful_barrier_count"))
            reported_rate = float(item.get("successful_barrier_rate"))
            flight_phase = str(item.get("flight_phase", "")).strip()
            aircraft_operator = str(item.get("aircraft_operator", "")).strip()
            if (
                candidate_count < 20
                or success_count < 0
                or success_count > candidate_count
                or not 0.0 <= reported_rate <= 1.0
                or not flight_phase
                or not aircraft_operator
            ):
                raise ValueError(f"{TASK_ID}: invalid aggregate row")
            rate = success_count / candidate_count
            if abs(reported_rate - rate) > 1e-6:
                raise ValueError(f"{TASK_ID}: inconsistent aggregate rate")
            projected_rows.append(
                {
                    "flight_phase": flight_phase,
                    "aircraft_operator": aircraft_operator,
                    "candidate_incident_count": candidate_count,
                    "successful_barrier_count": success_count,
                    "successful_barrier_rate": rate,
                }
            )
        if not projected_rows:
            raise ValueError(f"{TASK_ID}: aggregate returned no qualifying rows")
        all_projected = pd.DataFrame.from_records(projected_rows)
        if all_projected.duplicated(
            ["flight_phase", "aircraft_operator"]
        ).any():
            raise ValueError(f"{TASK_ID}: duplicate aggregate groups")
        projected = (
            all_projected
            .sort_values(
                [
                    "successful_barrier_rate",
                    "successful_barrier_count",
                    "flight_phase",
                    "aircraft_operator",
                ],
                ascending=[False, False, True, True],
                kind="stable",
            )
            .head(5)
            .reset_index(drop=True)
        )
        projected.insert(0, "rank", range(1, len(projected) + 1))
        tracker.record("project", len(aggregate_frame), projected)
        answer = df_records(projected)

    save_output(
        TASK_ID,
        answer,
        timer.elapsed + aggregate.resumed_elapsed_seconds,
        tracker,
    )


if __name__ == "__main__":
    main()
