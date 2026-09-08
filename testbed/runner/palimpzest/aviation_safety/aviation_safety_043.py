#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-043."""

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
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-043"
WEATHER_MECHANISMS = (
    "convective_weather",
    "turbulence",
    "windshear",
    "icing",
    "low_visibility",
    "low_ceiling",
    "strong_or_gusty_wind",
    "precipitation",
)
RECORDED_DECISIONS = (
    "diversion",
    "emergency_declaration",
    "new_clearance",
)
RESULT_TO_DECISION = {
    "diversion": "Flight Crew Diverted",
    "emergency_declaration": "General Declared Emergency",
    "new_clearance": "Air Traffic Control Issued New Clearance",
}


def supports_recorded_decision(row) -> bool:
    expected_label = RESULT_TO_DECISION.get(row["recorded_decision"])
    return expected_label in row["recorded_result_labels"]


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "local_time_of_day"],
        )
        tracker.record("scan", None, incidents)

        afternoon_incidents = incidents.loc[
            incidents["local_time_of_day"] == "1201-1800"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), afternoon_incidents)

        reports = load_selected_texts("asrs", afternoon_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(afternoon_incidents), reports)

        causal_plan = memory_dataset(
            f"{TASK_ID}-causal-weather", reports
        ).sem_filter(
            filter=(
                "Weather or visibility explicitly caused an operational decision, "
                "rather than merely co-occurring with it."
            ),
            depends_on=["text"],
        )
        started = time.time()
        causal_result = causal_plan.run(config)
        causal_reports = result_frame(causal_result, reports)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            causal_reports,
            causal_result,
            time.time() - started,
        )

        qualifying_plan = memory_dataset(
            f"{TASK_ID}-qualifying-weather", causal_reports
        ).sem_filter(
            filter=(
                "The weather mention is not incidental, and any turbulence "
                "described did not affect passenger comfort only."
            ),
            depends_on=["text"],
        )
        started = time.time()
        qualifying_result = qualifying_plan.run(config)
        qualifying_reports = result_frame(
            qualifying_result,
            causal_reports,
        )
        tracker.record_semantic(
            "sem_filter",
            len(causal_reports),
            qualifying_reports,
            qualifying_result,
            time.time() - started,
        )

        pair_plan = memory_dataset(
            f"{TASK_ID}-weather-decision-pairs", qualifying_reports
        ).sem_flat_map(
            cols=[
                {
                    "name": "weather_mechanism",
                    "type": str,
                    "desc": (
                        "Exactly one of convective_weather, turbulence, windshear, "
                        "icing, low_visibility, low_ceiling, "
                        "strong_or_gusty_wind, or precipitation."
                    ),
                },
                {
                    "name": "recorded_decision",
                    "type": str,
                    "desc": (
                        "Exactly one of diversion, emergency_declaration, or "
                        "new_clearance."
                    ),
                },
            ],
            desc=(
                "Emit one row for every explicitly supported causal pairing of "
                "a weather mechanism and operational decision in the report."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        pair_result = pair_plan.run(config)
        semantic_pairs = result_frame(
            pair_result,
            qualifying_reports,
            ["weather_mechanism", "recorded_decision"],
        )
        semantic_pairs["weather_mechanism"] = semantic_pairs[
            "weather_mechanism"
        ].map(lambda value: normalize_enum(value, WEATHER_MECHANISMS))
        semantic_pairs["recorded_decision"] = semantic_pairs[
            "recorded_decision"
        ].map(lambda value: normalize_enum(value, RECORDED_DECISIONS))
        tracker.record_semantic(
            "sem_flat_map",
            len(qualifying_reports),
            semantic_pairs,
            pair_result,
            time.time() - started,
        )

        semantic_pairs = semantic_pairs.loc[
            semantic_pairs["weather_mechanism"].notna()
            & semantic_pairs["recorded_decision"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(pair_result), semantic_pairs)

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        result_rows = events.loc[
            (events["event_type"] == "result")
            & events["label"].isin(RESULT_TO_DECISION.values())
        ].reset_index(drop=True)
        tracker.record("filter", len(events), result_rows)

        recorded_results = (
            result_rows.groupby("incident_id", sort=False)
            .agg(recorded_result_labels=("label", list))
            .reset_index()
        )
        tracker.record("groupby", len(result_rows), recorded_results)

        join_candidates = semantic_pairs.merge(
            recorded_results,
            on="incident_id",
            how="inner",
            validate="many_to_one",
        )
        if join_candidates.empty:
            joined = join_candidates.copy()
        else:
            joined = join_candidates.loc[
                join_candidates.apply(supports_recorded_decision, axis=1)
            ].reset_index(drop=True)
        tracker.record(
            "join",
            {"left": len(semantic_pairs), "right": len(recorded_results)},
            joined,
        )

        grouped = (
            joined.groupby(
                ["weather_mechanism", "recorded_decision"], sort=False
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record("groupby", len(joined), grouped)

        ordered = grouped.sort_values(
            ["incident_count", "weather_mechanism", "recorded_decision"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("sort", len(grouped), ordered)

        limited = ordered.head(20).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited.copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
