#!/usr/bin/env python3
"""Plan-optimization pipeline for aviation_safety-039."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

import palimpzest as pz  # noqa: E402
from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    normalize_enum,
    run_plan_optimization,
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


def normalize_issue(record: dict) -> dict:
    return {
        "normalized_documentation_issue": normalize_enum(
            record.get("documentation_issue_type"),
            DOCUMENTATION_TYPES,
        )
    }


def has_valid_issue(record: dict) -> bool:
    return bool(record.get("normalized_documentation_issue")) and bool(
        str(record.get("affected_item") or "").strip()
    )


def affected_item_mode(value) -> str | None:
    values = value if isinstance(value, list) else []
    return stable_mode(pd.Series(values, dtype="object")) if values else None


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem"],
        )
        aircraft = incidents.loc[
            incidents["primary_problem"] == "Aircraft"
        ].reset_index(drop=True)
        reports = load_selected_texts("asrs", aircraft)[["incident_id", "text"]]

        attributes = load_table(
            "asrs",
            "aircraft_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        status_rows = attributes.loc[
            attributes["attribute"].isin(STATUS_ATTRIBUTES)
        ].reset_index(drop=True)
        status = (
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

        components = load_table(
            "asrs",
            "components.csv",
            ["incident_id", "attribute", "value"],
        )
        component_rows = components.loc[
            components["attribute"].isin(["aircraft_component", "problem"])
        ].reset_index(drop=True)
        component_records = (
            component_rows.groupby("incident_id", sort=False)
            .agg(
                structured_components=(
                    "value",
                    lambda values: collect_for_attribute(
                        values,
                        component_rows,
                        "aircraft_component",
                    ),
                ),
                structured_problems=(
                    "value",
                    lambda values: collect_for_attribute(
                        values,
                        component_rows,
                        "problem",
                    ),
                ),
            )
            .reset_index()
        )

        issues = memory_dataset(f"{TASK_ID}-reports", reports).sem_filter(
            (
                "Keep this report only if it describes an actual maintenance-"
                "documentation problem affecting a component or required item. "
                "Exclude a mechanical discrepancy alone."
            ),
            depends_on=["text"],
        )
        issues = issues.sem_map(
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
                    "desc": "The affected component or item in at most six words.",
                },
            ],
            desc="Classify the documentation issue and extract its affected item.",
            depends_on=["incident_id", "text"],
        )
        issues = issues.map(
            normalize_issue,
            cols=[
                {
                    "name": "normalized_documentation_issue",
                    "type": str | None,
                    "desc": "Validated documentation issue label.",
                }
            ],
            depends_on=["documentation_issue_type"],
        )
        issues = issues.filter(
            has_valid_issue,
            depends_on=["normalized_documentation_issue", "affected_item"],
        )
        joined = issues.join(
            memory_dataset(f"{TASK_ID}-status", status),
            on="incident_id",
            how="inner",
        )
        joined = joined.join(
            memory_dataset(f"{TASK_ID}-components", component_records),
            on="incident_id",
            how="inner",
        )
        joined = joined.distinct(
            [
                "incident_id",
                "normalized_documentation_issue",
                "records_complete",
                "correct_document_on_board",
                "affected_item",
            ]
        )
        plan = joined.groupby(
            pz.GroupBySig(
                group_by_fields=[
                    "normalized_documentation_issue",
                    "records_complete",
                    "correct_document_on_board",
                ],
                agg_funcs=["count", "list"],
                agg_fields=["incident_id", "affected_item"],
            )
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True).rename(
            columns={
                "normalized_documentation_issue": "documentation_issue_type",
                "count(incident_id)": "incident_count",
                "list(affected_item)": "affected_items",
            }
        )
        output = output.reindex(
            columns=[
                "documentation_issue_type",
                "records_complete",
                "correct_document_on_board",
                "incident_count",
                "affected_items",
            ]
        )
        output["canonical_affected_item"] = output["affected_items"].map(
            affected_item_mode
        )
        tracker.record_semantic(
            "optimized_plan",
            {
                "reports": len(reports),
                "status": len(status),
                "components": len(component_records),
            },
            output,
            optimized.result,
            time.time() - started,
        )
        ordered = output.sort_values(
            [
                "incident_count",
                "documentation_issue_type",
                "records_complete",
                "correct_document_on_board",
            ],
            ascending=[False, True, True, True],
            kind="stable",
        ).head(15).reset_index(drop=True)
        ordered.insert(0, "rank", range(1, len(ordered) + 1))
        answer = df_records(
            ordered[
                [
                    "rank",
                    "documentation_issue_type",
                    "records_complete",
                    "correct_document_on_board",
                    "incident_count",
                    "canonical_affected_item",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
