#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-016."""

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

TASK_ID = "aviation_safety-016"
COMMUNICATION_PAIRS = ("flight_crew_atc", "flight_crew_flight_crew")
BREAKDOWN_MODES = (
    "phraseology",
    "timing_or_late_information",
    "assumption_or_expectation",
    "task_division_or_role",
    "failure_to_challenge",
)


def supports_pair(records, communication_pair: str) -> bool:
    for record in records:
        text = str(record)
        if communication_pair == "flight_crew_atc":
            if (
                "Party1 Flight Crew" in text and "Party2 ATC" in text
            ) or (
                "Party1 ATC" in text and "Party2 Flight Crew" in text
            ):
                return True
        elif (
            communication_pair == "flight_crew_flight_crew"
            and "Party1 Flight Crew" in text
            and "Party2 Flight Crew" in text
        ):
            return True
    return False


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem"],
        )
        tracker.record("scan", None, incidents)

        human_factors = incidents.loc[
            incidents["primary_problem"] == "Human Factors"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), human_factors)

        reports = load_selected_texts("asrs", human_factors)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(human_factors), reports)

        pair_plan = memory_dataset(TASK_ID, reports).sem_flat_map(
            cols=[
                {
                    "name": "communication_pair",
                    "type": str,
                    "desc": (
                        "Exactly one of flight_crew_atc or "
                        "flight_crew_flight_crew."
                    ),
                },
                {
                    "name": "breakdown_mode",
                    "type": str,
                    "desc": (
                        "The single primary mode for this participant group, "
                        "exactly one of phraseology, timing_or_late_information, "
                        "assumption_or_expectation, task_division_or_role, or "
                        "failure_to_challenge."
                    ),
                },
            ],
            desc=(
                "Emit one row for every clearly supported participant group in "
                "the report. For each emitted group, select one clearly primary "
                "communication-breakdown mode. Do not emit a group unless the "
                "narrative clearly supports both its participants and primary "
                "mode. Emit at most one row per participant group."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        pair_result = pair_plan.run(config)
        extracted = result_frame(pair_result)
        extracted["communication_pair"] = extracted[
            "communication_pair"
        ].map(lambda value: normalize_enum(value, COMMUNICATION_PAIRS))
        extracted["breakdown_mode"] = extracted["breakdown_mode"].map(
            lambda value: normalize_enum(value, BREAKDOWN_MODES)
        )
        tracker.record_semantic(
            "sem_flat_map",
            len(reports),
            extracted,
            pair_result,
            time.time() - started,
        )

        semantic_pairs = extracted.loc[
            extracted["communication_pair"].notna()
            & extracted["breakdown_mode"].notna(),
            ["incident_id", "communication_pair", "breakdown_mode"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), semantic_pairs)

        person_factors = load_table(
            "asrs",
            "person_factors.csv",
            ["incident_id", "factor_type", "value"],
        )
        tracker.record("scan", None, person_factors)

        communication_rows = person_factors.loc[
            (person_factors["factor_type"] == "communication_breakdown")
            & person_factors["value"].str.contains("Flight Crew", na=False)
        ].reset_index(drop=True)
        tracker.record("filter", len(person_factors), communication_rows)

        communication_pair_records = (
            communication_rows.groupby("incident_id", sort=False)
            .agg(
                communication_pair_records=(
                    "value",
                    lambda values: list(dict.fromkeys(values)),
                )
            )
            .reset_index()
        )
        tracker.record(
            "groupby", len(communication_rows), communication_pair_records
        )

        join_candidates = semantic_pairs.merge(
            communication_pair_records,
            on="incident_id",
            how="inner",
            validate="many_to_one",
        )
        joined = join_candidates.loc[
            join_candidates.apply(
                lambda row: supports_pair(
                    row["communication_pair_records"],
                    row["communication_pair"],
                ),
                axis=1,
            )
        ].reset_index(drop=True)
        tracker.record(
            "join",
            {
                "left": len(semantic_pairs),
                "right": len(communication_pair_records),
            },
            joined,
        )

        grouped = (
            joined.groupby(
                ["communication_pair", "breakdown_mode"], sort=True
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record("groupby", len(joined), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
