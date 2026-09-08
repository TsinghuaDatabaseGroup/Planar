#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-038."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "aviation_safety-038"
FACTOR_LABELS = {
    "Communication Breakdown": "communication_breakdown",
    "Workload": "workload",
    "Confusion": "confusion",
    "Situational Awareness": "situational_awareness",
}
MATCH_COLUMNS = [
    "incident_id",
    "narrative_factor_phrase",
    "actor",
    "operational_effect",
    "incident_factor_value",
    "factor_vocabulary",
    "canonical_human_factor",
]


def contains_target_factor(value) -> bool:
    text = str(value)
    return any(label in text for label in FACTOR_LABELS)


def represented_factors(value) -> list[str]:
    text = str(value)
    return [
        canonical
        for label, canonical in FACTOR_LABELS.items()
        if label in text
    ]


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

        human_factor_incidents = incidents.loc[
            incidents["primary_problem"] == "Human Factors"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), human_factor_incidents)

        reports = load_selected_texts("asrs", human_factor_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(human_factor_incidents), reports)

        extraction_plan = memory_dataset(TASK_ID, reports).sem_flat_map(
            cols=[
                {
                    "name": "narrative_factor_phrase",
                    "type": str,
                    "desc": (
                        "A human-factor phrase explicitly portrayed as contributing "
                        "to the event, in no more than six words."
                    ),
                },
                {
                    "name": "actor",
                    "type": str,
                    "desc": "The associated actor in no more than six words.",
                },
                {
                    "name": "operational_effect",
                    "type": str,
                    "desc": (
                        "The resulting operational effect in no more than six words."
                    ),
                },
            ],
            desc=(
                "Emit one row for each human-factor phrase that the narrative "
                "explicitly portrays as contributing to the event. Do not emit "
                "factors mentioned only as background."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted_factors = result_frame(
            extraction_result,
            reports,
            [
                "narrative_factor_phrase",
                "actor",
                "operational_effect",
            ],
        )
        tracker.record_semantic(
            "sem_flat_map",
            len(reports),
            extracted_factors,
            extraction_result,
            time.time() - started,
        )

        person_factors = load_table(
            "asrs",
            "person_factors.csv",
            ["incident_id", "factor_type", "value"],
        )
        tracker.record("scan", None, person_factors)

        incident_factor_records = person_factors.loc[
            (person_factors["factor_type"] == "human_factors")
            & person_factors["value"].map(contains_target_factor),
            ["incident_id", "value"],
        ].rename(columns={"value": "incident_factor_value"})
        incident_factor_records = incident_factor_records.reset_index(drop=True)
        tracker.record("filter", len(person_factors), incident_factor_records)

        factor_candidates = extracted_factors.merge(
            incident_factor_records,
            on="incident_id",
            how="inner",
            validate="many_to_many",
        )
        tracker.record(
            "join",
            {
                "left": len(extracted_factors),
                "right": len(incident_factor_records),
            },
            factor_candidates,
        )

        vocabulary_rows = person_factors.loc[
            (person_factors["factor_type"] == "human_factors")
            & person_factors["value"].map(contains_target_factor),
            ["value"],
        ].reset_index(drop=True)
        tracker.record("filter", len(person_factors), vocabulary_rows)

        distinct_vocabulary = vocabulary_rows.drop_duplicates(
            subset=["value"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(vocabulary_rows), distinct_vocabulary)

        factor_vocabulary = distinct_vocabulary.rename(
            columns={"value": "factor_vocabulary"}
        ).copy()
        factor_vocabulary["canonical_human_factor"] = factor_vocabulary[
            "factor_vocabulary"
        ].map(represented_factors)
        factor_vocabulary = factor_vocabulary.explode(
            "canonical_human_factor", ignore_index=True
        )
        tracker.record("project", len(distinct_vocabulary), factor_vocabulary)

        left_factors = memory_dataset(
            f"{TASK_ID}-factor-candidates", factor_candidates
        )
        right_factors = memory_dataset(
            f"{TASK_ID}-factor-vocabulary", factor_vocabulary
        )
        match_plan = left_factors.sem_join(
            right_factors,
            condition=(
                "The narrative factor phrase semantically denotes exactly the "
                "canonical human factor represented by the vocabulary row, and "
                "that canonical factor is also compatible with the joined "
                "incident's structured factor value. Exact wording is not "
                "required; the vocabulary is global and generic topical "
                "similarity is insufficient."
            ),
            depends_on=MATCH_COLUMNS,
        )
        started = time.time()
        match_result = match_plan.run(config)
        semantic_factor_matches = result_frame(match_result)
        if semantic_factor_matches.empty:
            semantic_factor_matches = pd.DataFrame(columns=MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {
                "left": len(factor_candidates),
                "right": len(factor_vocabulary),
            },
            semantic_factor_matches,
            match_result,
            time.time() - started,
        )

        canonical_matches = semantic_factor_matches.drop_duplicates(
            subset=[
                "incident_id",
                "narrative_factor_phrase",
                "actor",
                "operational_effect",
                "canonical_human_factor",
            ]
        ).reset_index(drop=True)
        tracker.record(
            "distinct", len(semantic_factor_matches), canonical_matches
        )

        grouped = (
            canonical_matches.groupby("canonical_human_factor", sort=True)
            .agg(
                incident_count=("incident_id", "nunique"),
                most_common_actor=("actor", stable_mode),
                most_common_operational_effect=(
                    "operational_effect",
                    stable_mode,
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(canonical_matches), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
