#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-036."""

from __future__ import annotations

import os
import re
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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-036"
EXPERIENCE_ROLES = ("causal", "incidental", "not_mentioned")
PERSON_ATTRIBUTES = ("function", "qualification", "location_of_person")


def parse_experience_value(value, label: str) -> float | None:
    match = re.search(
        rf"(?:^|;\s*){re.escape(label)}\s+(-?\d+(?:\.\d+)?)",
        str(value),
    )
    return float(match.group(1)) if match else None


def max_for_attribute(
    values: pd.Series,
    frame: pd.DataFrame,
    attribute: str,
):
    selected = values.loc[
        frame.loc[values.index, "attribute"] == attribute
    ].dropna()
    return selected.max() if not selected.empty else pd.NA


def normalize_crew_function(value) -> str:
    function = "" if pd.isna(value) else str(value)
    if "Captain" in function:
        return "captain"
    if "First Officer" in function:
        return "first_officer"
    if "Single Pilot" in function:
        return "single_pilot"
    if "Instructor" in function:
        return "instructor"
    return "other_flight_crew"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        person_factors = load_table(
            "asrs",
            "person_factors.csv",
            ["incident_id", "person_ref", "factor_type", "value"],
        )
        tracker.record("scan", None, person_factors)

        experience_rows = person_factors.loc[
            (person_factors["factor_type"] == "experience")
            & person_factors["value"].str.contains("Flight Crew", na=False)
        ].reset_index(drop=True)
        tracker.record("filter", len(person_factors), experience_rows)

        experience_rows = experience_rows[
            ["incident_id", "person_ref", "value"]
        ].copy()
        experience_rows["total_hours"] = experience_rows["value"].map(
            lambda value: parse_experience_value(value, "Flight Crew Total")
        )
        experience_rows["type_hours"] = experience_rows["value"].map(
            lambda value: parse_experience_value(value, "Flight Crew Type")
        )
        experience_rows["last_90_days_hours"] = experience_rows["value"].map(
            lambda value: parse_experience_value(
                value, "Flight Crew Last 90 Days"
            )
        )
        experience_rows = experience_rows[
            [
                "incident_id",
                "person_ref",
                "total_hours",
                "type_hours",
                "last_90_days_hours",
            ]
        ]
        tracker.record("project", len(experience_rows), experience_rows)

        person_attributes = load_table(
            "asrs",
            "person_attributes.csv",
            ["incident_id", "person_ref", "attribute", "value"],
        )
        tracker.record("scan", None, person_attributes)

        attribute_rows = person_attributes.loc[
            person_attributes["attribute"].isin(PERSON_ATTRIBUTES)
        ].reset_index(drop=True)
        tracker.record("filter", len(person_attributes), attribute_rows)

        person_rows = (
            attribute_rows.groupby(
                ["incident_id", "person_ref"], sort=False
            )
            .agg(
                function_value=(
                    "value",
                    lambda values: max_for_attribute(
                        values, attribute_rows, "function"
                    ),
                ),
                qualification=(
                    "value",
                    lambda values: max_for_attribute(
                        values, attribute_rows, "qualification"
                    ),
                ),
                location_of_person=(
                    "value",
                    lambda values: max_for_attribute(
                        values, attribute_rows, "location_of_person"
                    ),
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(attribute_rows), person_rows)

        person_experience = experience_rows.merge(
            person_rows,
            on=["incident_id", "person_ref"],
            how="inner",
            validate="many_to_one",
        )
        tracker.record(
            "join",
            {"left": len(experience_rows), "right": len(person_rows)},
            person_experience,
        )

        structured_people = person_experience.copy()
        structured_people["crew_function"] = structured_people[
            "function_value"
        ].map(normalize_crew_function)
        structured_people = structured_people[
            [
                "incident_id",
                "person_ref",
                "crew_function",
                "total_hours",
                "type_hours",
                "last_90_days_hours",
                "qualification",
                "location_of_person",
            ]
        ]
        tracker.record("project", len(person_experience), structured_people)

        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "primary_problem", "text_file"],
        )
        tracker.record("scan", None, incidents)

        target_incidents = incidents.loc[
            incidents["primary_problem"].isin(["Human Factors", "Procedure"])
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), target_incidents)

        joined = structured_people.merge(
            target_incidents,
            on="incident_id",
            how="inner",
            validate="many_to_one",
        )
        tracker.record(
            "join",
            {"left": len(structured_people), "right": len(target_incidents)},
            joined,
        )

        reports = load_selected_texts("asrs", joined)[
            [
                "incident_id",
                "person_ref",
                "crew_function",
                "total_hours",
                "type_hours",
                "last_90_days_hours",
                "qualification",
                "location_of_person",
                "text",
            ]
        ]
        tracker.record("scan", len(joined), reports)

        role_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "experience_role",
                    "type": str,
                    "desc": (
                        "Exactly one of causal, incidental, or not_mentioned. "
                        "Use causal when this person's low or recent experience, "
                        "training, qualification, or unfamiliarity contributed; "
                        "incidental when experience is only context; and "
                        "not_mentioned when this person's experience is absent "
                        "from the narrative."
                    ),
                }
            ],
            desc=(
                "Classify the narrative role of each structured person's "
                "experience. Use crew function, qualification, and location to "
                "resolve which person the narrative discusses."
            ),
            depends_on=[
                "incident_id",
                "person_ref",
                "crew_function",
                "qualification",
                "location_of_person",
                "text",
            ],
        )
        started = time.time()
        role_result = role_plan.run(config)
        classified = result_frame(role_result)
        classified["experience_role"] = classified["experience_role"].map(
            lambda value: normalize_enum(value, EXPERIENCE_ROLES)
        )
        tracker.record_semantic(
            "sem_map",
            len(reports),
            classified,
            role_result,
            time.time() - started,
        )

        selected = classified.loc[
            classified["experience_role"].notna()
        ].reset_index(drop=True)
        selected["person_key"] = list(
            zip(selected["incident_id"], selected["person_ref"])
        )
        tracker.record("filter", len(classified), selected)

        grouped = (
            selected.groupby(["crew_function", "experience_role"], sort=True)
            .agg(
                person_count=("person_key", "nunique"),
                median_total_hours=("total_hours", "median"),
                median_type_hours=("type_hours", "median"),
                median_last_90_days_hours=("last_90_days_hours", "median"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(selected), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
