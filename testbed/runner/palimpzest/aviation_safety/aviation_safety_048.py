#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-048."""

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
)

TASK_ID = "aviation_safety-048"
RESULT_LABELS = (
    "General Flight Cancelled / Delayed",
    "Flight Crew Returned To Departure Airport",
    "General Maintenance Action",
)
STATUS_ATTRIBUTES = (
    "maintenance_status_maintenance_deferred",
    "maintenance_status_released_for_service",
    "maintenance_status_records_complete",
    "maintenance_status_required_correct_doc_on_board",
)
DRIVER_CATEGORIES = (
    "component_discrepancy_or_failure",
    "mel_or_deferred_item",
    "troubleshooting_or_repair",
    "documentation_or_release_to_service",
    "required_inspection",
)
JOIN_INPUT_COLUMNS = [
    "incident_id",
    "narrative_affected_item",
    "maintenance_driver_phrase",
    "recorded_results",
    "canonical_component",
]
MATCH_COLUMNS = [*JOIN_INPUT_COLUMNS, "text"]


def set_union(values: pd.Series) -> list[str]:
    union = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            union.update(str(item) for item in value)
    return sorted(union)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        result_rows = events.loc[
            (events["event_type"] == "result")
            & events["label"].isin(RESULT_LABELS)
        ].reset_index(drop=True)
        tracker.record("filter", len(events), result_rows)

        recorded_results = (
            result_rows.groupby("incident_id", sort=False)
            .agg(recorded_results=("label", list))
            .reset_index()
        )
        tracker.record("groupby", len(result_rows), recorded_results)

        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file"],
        )
        tracker.record("scan", None, incidents)

        result_incidents = recorded_results.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {"left": len(recorded_results), "right": len(incidents)},
            result_incidents,
        )

        reports = load_selected_texts("asrs", result_incidents)[
            ["incident_id", "recorded_results", "text"]
        ]
        tracker.record("scan", len(result_incidents), reports)

        maintenance_plan = memory_dataset(
            f"{TASK_ID}-maintenance-driver", reports
        ).sem_filter(
            filter=(
                "Maintenance, rather than weather, crew legality, or passenger "
                "issues, was the primary driver of the recorded disruption."
            ),
            depends_on=["text"],
        )
        started = time.time()
        maintenance_result = maintenance_plan.run(config)
        maintenance_reports = result_frame(maintenance_result, reports)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            maintenance_reports,
            maintenance_result,
            time.time() - started,
        )

        affected_item_plan = memory_dataset(
            f"{TASK_ID}-affected-item", maintenance_reports
        ).sem_filter(
            filter=(
                "A specific affected component or required maintenance item can "
                "be identified from the report."
            ),
            depends_on=["text"],
        )
        started = time.time()
        affected_item_result = affected_item_plan.run(config)
        affected_item_reports = result_frame(
            affected_item_result,
            maintenance_reports,
        )
        tracker.record_semantic(
            "sem_filter",
            len(maintenance_reports),
            affected_item_reports,
            affected_item_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-maintenance-details", affected_item_reports
        ).sem_map(
            cols=[
                {
                    "name": "narrative_affected_item",
                    "type": str,
                    "desc": "The affected item phrase used in the narrative.",
                },
                {
                    "name": "maintenance_driver_phrase",
                    "type": str,
                    "desc": "A concise phrase describing the maintenance driver.",
                },
            ],
            desc=(
                "Extract the narrative affected-item phrase and a concise "
                "maintenance-driver phrase."
            ),
            depends_on=["incident_id", "recorded_results", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        maintenance_candidates = result_frame(
            extraction_result,
            affected_item_reports,
            ["narrative_affected_item", "maintenance_driver_phrase"],
        )
        tracker.record_semantic(
            "sem_map",
            len(affected_item_reports),
            maintenance_candidates,
            extraction_result,
            time.time() - started,
        )

        components = load_table(
            "asrs",
            "components.csv",
            ["attribute", "value"],
        )
        tracker.record("scan", None, components)

        vocabulary_rows = components.loc[
            components["attribute"] == "aircraft_component", ["value"]
        ].reset_index(drop=True)
        tracker.record("filter", len(components), vocabulary_rows)

        component_vocabulary = (
            vocabulary_rows.drop_duplicates(subset=["value"])
            .rename(columns={"value": "canonical_component"})
            .reset_index(drop=True)
        )
        tracker.record("distinct", len(vocabulary_rows), component_vocabulary)

        left_items = memory_dataset(
            f"{TASK_ID}-maintenance-candidates", maintenance_candidates
        )
        right_items = memory_dataset(
            f"{TASK_ID}-component-vocabulary", component_vocabulary
        )
        match_plan = left_items.sem_join(
            right_items,
            condition=(
                "The narrative affected item and canonical component label are "
                "synonyms, abbreviations, or have a clear part-to-system "
                "relationship. Do not use incident ID, exact spelling alone, or "
                "generic topical similarity."
            ),
            depends_on=JOIN_INPUT_COLUMNS,
        )
        started = time.time()
        match_result = match_plan.run(config)
        semantic_component_matches = result_frame(match_result)
        if semantic_component_matches.empty:
            semantic_component_matches = pd.DataFrame(columns=MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {
                "left": len(maintenance_candidates),
                "right": len(component_vocabulary),
            },
            semantic_component_matches,
            match_result,
            time.time() - started,
        )

        aircraft_attributes = load_table(
            "asrs",
            "aircraft_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, aircraft_attributes)

        status_rows = aircraft_attributes.loc[
            aircraft_attributes["attribute"].isin(STATUS_ATTRIBUTES)
        ].copy()
        status_rows["status_record"] = list(
            zip(status_rows["attribute"], status_rows["value"])
        )
        status_rows = status_rows.reset_index(drop=True)
        tracker.record("filter", len(aircraft_attributes), status_rows)

        maintenance_status_records = (
            status_rows.groupby("incident_id", sort=False)
            .agg(maintenance_status_records=("status_record", list))
            .reset_index()
        )
        tracker.record("groupby", len(status_rows), maintenance_status_records)

        joined = semantic_component_matches.merge(
            maintenance_status_records,
            on="incident_id",
            how="left",
            validate="many_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(semantic_component_matches),
                "right": len(maintenance_status_records),
            },
            joined,
        )

        driver_plan = memory_dataset(
            f"{TASK_ID}-driver-category", joined
        ).sem_map(
            cols=[
                {
                    "name": "maintenance_driver",
                    "type": str,
                    "desc": (
                        "Exactly one of component_discrepancy_or_failure, "
                        "mel_or_deferred_item, troubleshooting_or_repair, "
                        "documentation_or_release_to_service, or "
                        "required_inspection."
                    ),
                }
            ],
            desc="Assign the maintenance-driver category.",
            depends_on=[
                "maintenance_driver_phrase",
                "recorded_results",
                "maintenance_status_records",
                "text",
            ],
        )
        started = time.time()
        driver_result = driver_plan.run(config)
        classified = result_frame(
            driver_result,
            joined,
            ["maintenance_driver"],
        )
        if classified.empty:
            classified = joined.copy()
            classified["maintenance_driver"] = pd.Series(dtype="object")
        else:
            classified["maintenance_driver"] = classified[
                "maintenance_driver"
            ].map(lambda value: normalize_enum(value, DRIVER_CATEGORIES))
        tracker.record_semantic(
            "sem_map",
            len(joined),
            classified,
            driver_result,
            time.time() - started,
        )

        classified = classified.loc[
            classified["maintenance_driver"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(driver_result), classified)

        grouped = (
            classified.groupby(
                ["maintenance_driver", "canonical_component"], sort=False
            )
            .agg(
                incident_count=("incident_id", "nunique"),
                recorded_result_labels=("recorded_results", set_union),
            )
            .reset_index()
        )
        tracker.record("groupby", len(classified), grouped)

        ordered = grouped.sort_values(
            ["incident_count", "maintenance_driver", "canonical_component"],
            ascending=[False, True, True],
            kind="stable",
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
