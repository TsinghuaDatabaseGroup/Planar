#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-039."""

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
    normalize_enum,
    result_frame,
    save_output,
    stable_mode,
)

TASK_ID = "aviation_safety-039"
DOCUMENTATION_TYPES = (
    "incomplete_or_inaccurate_logbook",
    "missing_or_incomplete_work_card",
    "incorrect_or_incomplete_mel_procedure",
    "missing_required_document",
    "release_or_signoff_error",
)
STATUS_ATTRIBUTES = (
    "maintenance_status_records_complete",
    "maintenance_status_required_correct_doc_on_board",
)


def max_for_attribute(
    values: pd.Series,
    frame: pd.DataFrame,
    attribute: str,
):
    selected = values.loc[
        frame.loc[values.index, "attribute"] == attribute
    ].dropna()
    return selected.max() if not selected.empty else pd.NA


def collect_for_attribute(
    values: pd.Series,
    frame: pd.DataFrame,
    attribute: str,
) -> list[str]:
    selected = values.loc[
        frame.loc[values.index, "attribute"] == attribute
    ].dropna()
    return list(dict.fromkeys(selected))


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

        aircraft = incidents.loc[
            incidents["primary_problem"] == "Aircraft"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), aircraft)

        reports = load_selected_texts("asrs", aircraft)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(aircraft), reports)

        documentation_plan = memory_dataset(
            f"{TASK_ID}-documentation", reports
        ).sem_filter(
            (
                "Keep this report only if it describes an actual maintenance-"
                "documentation problem that affected a component or required "
                "item. Exclude reports describing only a mechanical discrepancy "
                "without a documentation problem."
            ),
            depends_on=["text"],
        )
        started = time.time()
        documentation_result = documentation_plan.run(config)
        documentation_reports = result_frame(documentation_result)[
            ["incident_id", "text"]
        ]
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            documentation_reports,
            documentation_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", documentation_reports
        ).sem_map(
            cols=[
                {
                    "name": "documentation_issue_type",
                    "type": str,
                    "desc": (
                        "Exactly one of incomplete_or_inaccurate_logbook, "
                        "missing_or_incomplete_work_card, "
                        "incorrect_or_incomplete_mel_procedure, "
                        "missing_required_document, or release_or_signoff_error."
                    ),
                },
                {
                    "name": "affected_item",
                    "type": str,
                    "desc": (
                        "The component or required item affected by the "
                        "documentation problem, in no more than six words."
                    ),
                },
            ],
            desc=(
                "Classify the primary maintenance-documentation issue and extract "
                "a short canonical phrase for the affected item."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(extraction_result)
        extracted["documentation_issue_type"] = extracted[
            "documentation_issue_type"
        ].map(lambda value: normalize_enum(value, DOCUMENTATION_TYPES))
        extracted["affected_item"] = (
            extracted["affected_item"].fillna("").astype(str).str.strip()
        )
        tracker.record_semantic(
            "sem_map",
            len(documentation_reports),
            extracted,
            extraction_result,
            time.time() - started,
        )

        documentation_issues = extracted.loc[
            extracted["documentation_issue_type"].notna()
            & extracted["affected_item"].ne(""),
            ["incident_id", "documentation_issue_type", "affected_item"],
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), documentation_issues)

        aircraft_attributes = load_table(
            "asrs",
            "aircraft_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, aircraft_attributes)

        status_rows = aircraft_attributes.loc[
            aircraft_attributes["attribute"].isin(STATUS_ATTRIBUTES)
        ].reset_index(drop=True)
        tracker.record("filter", len(aircraft_attributes), status_rows)

        documentation_status = (
            status_rows.groupby("incident_id", sort=False)
            .agg(
                records_complete=(
                    "value",
                    lambda values: max_for_attribute(
                        values,
                        status_rows,
                        "maintenance_status_records_complete",
                    ),
                ),
                correct_document_on_board=(
                    "value",
                    lambda values: max_for_attribute(
                        values,
                        status_rows,
                        "maintenance_status_required_correct_doc_on_board",
                    ),
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(status_rows), documentation_status)

        issues_with_status = documentation_issues.merge(
            documentation_status,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(documentation_issues),
                "right": len(documentation_status),
            },
            issues_with_status,
        )

        components = load_table(
            "asrs",
            "components.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, components)

        component_rows = components.loc[
            components["attribute"].isin(["aircraft_component", "problem"])
        ].reset_index(drop=True)
        tracker.record("filter", len(components), component_rows)

        component_records = (
            component_rows.groupby("incident_id", sort=False)
            .agg(
                structured_components=(
                    "value",
                    lambda values: collect_for_attribute(
                        values, component_rows, "aircraft_component"
                    ),
                ),
                structured_problems=(
                    "value",
                    lambda values: collect_for_attribute(
                        values, component_rows, "problem"
                    ),
                ),
            )
            .reset_index()
        )
        tracker.record("groupby", len(component_rows), component_records)

        joined = issues_with_status.merge(
            component_records,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(issues_with_status),
                "right": len(component_records),
            },
            joined,
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
        tracker.record("groupby", len(joined), grouped)

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
        tracker.record("sort", len(grouped), ordered)

        limited = ordered.head(15).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited.copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
