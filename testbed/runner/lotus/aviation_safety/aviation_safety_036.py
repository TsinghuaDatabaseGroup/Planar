#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-036."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    parse_number_after,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-036"
CREW_FUNCTIONS = (
    "captain",
    "first_officer",
    "single_pilot",
    "instructor",
    "other_flight_crew",
)
EXPERIENCE_ROLES = ("causal", "incidental", "not_mentioned")
PERSON_ATTRIBUTES = {"function", "qualification", "location_of_person"}


def maximum_value(group, attribute):
    values = group.loc[group["attribute"] == attribute, "value"].dropna().tolist()
    return max(values) if values else None


def crew_function(function_value) -> str:
    value = function_value if isinstance(function_value, str) else ""
    if "Captain" in value:
        return "captain"
    if "First Officer" in value:
        return "first_officer"
    if "Single Pilot" in value:
        return "single_pilot"
    if "Instructor" in value:
        return "instructor"
    return "other_flight_crew"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        person_factors = load_table("asrs", "person_factors.csv")[
            ["incident_id", "person_ref", "factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv)",
            None,
            len(person_factors),
            output=person_factors,
        )

        experience = person_factors[
            (person_factors["factor_type"] == "experience")
            & person_factors["value"].str.contains(
                "Flight Crew",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(factor_type='experience' AND value CONTAINS Flight Crew)",
            len(person_factors),
            len(experience),
            output=experience,
        )

        experience_rows = experience[["incident_id", "person_ref"]].copy()
        experience_rows["total_hours"] = experience["value"].map(
            lambda value: parse_number_after(value, "Flight Crew Total")
        )
        experience_rows["type_hours"] = experience["value"].map(
            lambda value: parse_number_after(value, "Flight Crew Type")
        )
        experience_rows["last_90_days_hours"] = experience["value"].map(
            lambda value: parse_number_after(value, "Flight Crew Last 90 Days")
        )
        tracker.record(
            "PROJECT(total_hours, type_hours, last_90_days_hours)",
            len(experience),
            len(experience_rows),
            output=experience_rows,
        )

        person_attributes = load_table("asrs", "person_attributes.csv")[
            ["incident_id", "person_ref", "attribute", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_attributes.csv)",
            None,
            len(person_attributes),
            output=person_attributes,
        )

        attribute_rows = person_attributes[
            person_attributes["attribute"].isin(PERSON_ATTRIBUTES)
        ].copy()
        tracker.record(
            "FILTER(attribute IN function/qualification/location_of_person)",
            len(person_attributes),
            len(attribute_rows),
            output=attribute_rows,
        )

        person_records = []
        for (incident_id, person_ref), group in attribute_rows.groupby(
            ["incident_id", "person_ref"],
            sort=False,
        ):
            person_records.append(
                {
                    "incident_id": incident_id,
                    "person_ref": person_ref,
                    "function_value": maximum_value(group, "function"),
                    "qualification": maximum_value(group, "qualification"),
                    "location_of_person": maximum_value(
                        group,
                        "location_of_person",
                    ),
                }
            )
        person_rows = pd.DataFrame.from_records(
            person_records,
            columns=[
                "incident_id",
                "person_ref",
                "function_value",
                "qualification",
                "location_of_person",
            ],
        )
        tracker.record(
            "GROUP_BY([incident_id, person_ref], MAX_IF(person attributes))",
            len(attribute_rows),
            len(person_rows),
            output=person_rows,
        )

        person_experience = experience_rows.merge(
            person_rows,
            on=["incident_id", "person_ref"],
            how="inner",
            validate="many_to_one",
        )
        tracker.record(
            "JOIN(experience_rows, person_rows, incident_id and person_ref)",
            {"left": len(experience_rows), "right": len(person_rows)},
            len(person_experience),
            output=person_experience,
        )

        structured_people = person_experience[
            [
                "incident_id",
                "person_ref",
                "total_hours",
                "type_hours",
                "last_90_days_hours",
                "qualification",
                "location_of_person",
            ]
        ].copy()
        structured_people["crew_function"] = person_experience[
            "function_value"
        ].map(crew_function)
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
        tracker.record(
            "PROJECT(crew_function=CASE and experience fields)",
            len(person_experience),
            len(structured_people),
            output=structured_people,
        )

        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "primary_problem", "text_file"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        target_incidents = incidents[
            incidents["primary_problem"].isin(["Human Factors", "Procedure"])
        ].copy()
        tracker.record(
            "FILTER(primary_problem IN Human Factors/Procedure)",
            len(incidents),
            len(target_incidents),
            output=target_incidents,
        )

        joined = structured_people.merge(
            target_incidents[["incident_id", "text_file"]],
            on="incident_id",
            how="inner",
            validate="many_to_one",
        )
        tracker.record(
            "JOIN(structured_people, incidents, incident_id)",
            {
                "left": len(structured_people),
                "right": len(target_incidents),
            },
            len(joined),
            output=joined,
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
        tracker.record(
            "SCAN_DOCS(selector=joined.text_file)",
            len(joined),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(classify each person's experience role)",
            input_rows=len(reports),
        ) as step:
            classified = reports.sem_extract(
                input_cols=[
                    "person_ref",
                    "crew_function",
                    "qualification",
                    "location_of_person",
                    "text",
                ],
                output_cols={
                    "experience_role": (
                        "classify this structured person's experience as causal when "
                        "low or recent experience, training, qualification, or "
                        "unfamiliarity contributed; incidental when mentioned only "
                        "as context; or not_mentioned when absent. Use the supplied "
                        "person function, qualification, and location to resolve "
                        "which narrative person is being assessed"
                    )
                },
            )
            classified["experience_role"] = classified[
                "experience_role"
            ].map(lambda value: normalize_enum(value, EXPERIENCE_ROLES))
            classified["experience_role"] = classified[
                "experience_role"
            ].fillna("not_mentioned")
            classified = classified[
                [
                    "incident_id",
                    "person_ref",
                    "crew_function",
                    "experience_role",
                    "total_hours",
                    "type_hours",
                    "last_90_days_hours",
                ]
            ].reset_index(drop=True)
            step.set_output(classified)

        classified["_person_key"] = (
            classified["incident_id"].astype(str)
            + "\x1f"
            + classified["person_ref"].astype(str)
        )
        grouped = (
            classified.groupby(
                ["crew_function", "experience_role"],
                sort=False,
            )
            .agg(
                person_count=("_person_key", "nunique"),
                median_total_hours=("total_hours", "median"),
                median_type_hours=("type_hours", "median"),
                median_last_90_days_hours=("last_90_days_hours", "median"),
            )
            .reset_index()
        )
        crew_order = {
            label: index for index, label in enumerate(CREW_FUNCTIONS)
        }
        role_order = {
            label: index for index, label in enumerate(EXPERIENCE_ROLES)
        }
        grouped = (
            grouped.assign(
                _crew_order=grouped["crew_function"].map(crew_order),
                _role_order=grouped["experience_role"].map(role_order),
            )
            .sort_values(["_crew_order", "_role_order"])
            .drop(columns=["_crew_order", "_role_order"])
            .reset_index(drop=True)
        )
        tracker.record(
            "GROUP_BY([crew_function, experience_role], COUNT_DISTINCT, MEDIAN)",
            len(classified),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
