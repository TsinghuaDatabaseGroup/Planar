#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-039."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    save_output,
    setup,
    stable_mode,
)

TASK_ID = "aviation_safety-039"
ISSUE_TYPES = (
    "incomplete_or_inaccurate_logbook",
    "missing_or_incomplete_work_card",
    "incorrect_or_incomplete_mel_procedure",
    "missing_required_document",
    "release_or_signoff_error",
)
STATUS_ATTRIBUTES = {
    "maintenance_status_records_complete",
    "maintenance_status_required_correct_doc_on_board",
}
COMPONENT_ATTRIBUTES = {"aircraft_component", "problem"}


def maximum_value(group, attribute):
    values = group.loc[group["attribute"] == attribute, "value"].dropna().tolist()
    return max(values) if values else None


def collected_values(group, attribute):
    return group.loc[group["attribute"] == attribute, "value"].dropna().tolist()


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "primary_problem"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        aircraft_incidents = incidents[
            incidents["primary_problem"] == "Aircraft"
        ].copy()
        tracker.record(
            "FILTER(primary_problem='Aircraft')",
            len(incidents),
            len(aircraft_incidents),
            output=aircraft_incidents,
        )

        reports = load_selected_texts("asrs", aircraft_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(aircraft_incidents),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(actual maintenance-documentation problem)",
            input_rows=len(reports),
        ) as step:
            documentation_reports = reports.sem_filter(
                "The report {text} describes an actual maintenance-documentation "
                "problem that affected a component or required item, rather than a "
                "mechanical discrepancy alone"
            )
            step.set_output(documentation_reports)

        with tracker.step(
            "SEM_EXTRACT(documentation issue type and affected item)",
            input_rows=len(documentation_reports),
        ) as step:
            classified = documentation_reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "documentation_issue_type": (
                        "exactly one of incomplete_or_inaccurate_logbook, "
                        "missing_or_incomplete_work_card, "
                        "incorrect_or_incomplete_mel_procedure, "
                        "missing_required_document, release_or_signoff_error"
                    ),
                    "affected_item": (
                        "the component or required item affected by the documentation "
                        "problem as a natural phrase of at most six words"
                    ),
                },
            )
            classified["documentation_issue_type"] = classified[
                "documentation_issue_type"
            ].map(lambda value: normalize_enum(value, ISSUE_TYPES))
            classified["affected_item"] = classified["affected_item"].map(
                clean_text
            )
            documentation_issues = classified.loc[
                classified["documentation_issue_type"].notna(),
                ["incident_id", "documentation_issue_type", "affected_item"],
            ].reset_index(drop=True)
            step.set_output(documentation_issues)

        aircraft_attributes = load_table("asrs", "aircraft_attributes.csv")[
            ["incident_id", "attribute", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(aircraft_attributes.csv)",
            None,
            len(aircraft_attributes),
            output=aircraft_attributes,
        )

        status_rows = aircraft_attributes[
            aircraft_attributes["attribute"].isin(STATUS_ATTRIBUTES)
        ].copy()
        tracker.record(
            "FILTER(attribute IN maintenance documentation statuses)",
            len(aircraft_attributes),
            len(status_rows),
            output=status_rows,
        )

        status_records = []
        for incident_id, group in status_rows.groupby("incident_id", sort=False):
            status_records.append(
                {
                    "incident_id": incident_id,
                    "records_complete": maximum_value(
                        group,
                        "maintenance_status_records_complete",
                    ),
                    "correct_document_on_board": maximum_value(
                        group,
                        "maintenance_status_required_correct_doc_on_board",
                    ),
                }
            )
        documentation_status = pd.DataFrame.from_records(
            status_records,
            columns=[
                "incident_id",
                "records_complete",
                "correct_document_on_board",
            ],
        )
        tracker.record(
            "GROUP_BY([incident_id], MAX_IF(documentation statuses))",
            len(status_rows),
            len(documentation_status),
            output=documentation_status,
        )

        issues_with_status = documentation_issues.merge(
            documentation_status,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(documentation_issues, documentation_status, incident_id)",
            {
                "left": len(documentation_issues),
                "right": len(documentation_status),
            },
            len(issues_with_status),
            output=issues_with_status,
        )

        components = load_table("asrs", "components.csv")[
            ["incident_id", "attribute", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(components.csv)",
            None,
            len(components),
            output=components,
        )

        component_rows = components[
            components["attribute"].isin(COMPONENT_ATTRIBUTES)
        ].copy()
        tracker.record(
            "FILTER(attribute IN aircraft_component/problem)",
            len(components),
            len(component_rows),
            output=component_rows,
        )

        structured_rows = []
        for incident_id, group in component_rows.groupby(
            "incident_id",
            sort=False,
        ):
            structured_rows.append(
                {
                    "incident_id": incident_id,
                    "structured_components": collected_values(
                        group,
                        "aircraft_component",
                    ),
                    "structured_problems": collected_values(group, "problem"),
                }
            )
        component_records = pd.DataFrame.from_records(
            structured_rows,
            columns=[
                "incident_id",
                "structured_components",
                "structured_problems",
            ],
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT_IF(component/problem))",
            len(component_rows),
            len(component_records),
            output=component_records,
        )

        joined = issues_with_status.merge(
            component_records,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(issues_with_status, component_records, incident_id)",
            {
                "left": len(issues_with_status),
                "right": len(component_records),
            },
            len(joined),
            output=joined,
        )

        grouped = (
            joined.groupby(
                [
                    "documentation_issue_type",
                    "records_complete",
                    "correct_document_on_board",
                ],
                sort=False,
                dropna=False,
            )
            .agg(
                incident_count=("incident_id", "nunique"),
                canonical_affected_item=("affected_item", stable_mode),
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([issue_type, records_complete, correct_document], "
            "COUNT_DISTINCT, MODE)",
            len(joined),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            [
                "incident_count",
                "documentation_issue_type",
                "records_complete",
                "correct_document_on_board",
            ],
            ascending=[False, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([count DESC, issue_type ASC, status values ASC])",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(15).copy()
        tracker.record(
            "LIMIT(15)",
            len(ordered),
            len(limited),
            output=limited,
        )

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and documentation status fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
