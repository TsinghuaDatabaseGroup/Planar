#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-048."""

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
)

TASK_ID = "aviation_safety-048"
TARGET_RESULTS = {
    "General Flight Cancelled / Delayed",
    "Flight Crew Returned To Departure Airport",
    "General Maintenance Action",
}
MAINTENANCE_STATUS_ATTRIBUTES = {
    "maintenance_status_maintenance_deferred",
    "maintenance_status_released_for_service",
    "maintenance_status_records_complete",
    "maintenance_status_required_correct_doc_on_board",
}
MAINTENANCE_DRIVERS = (
    "component_discrepancy_or_failure",
    "mel_or_deferred_item",
    "troubleshooting_or_repair",
    "documentation_or_release_to_service",
    "required_inspection",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        result_rows = events[
            (events["event_type"] == "result")
            & events["label"].isin(TARGET_RESULTS)
        ].copy()
        tracker.record(
            "FILTER(event_type='result' AND label IN target results)",
            len(events),
            len(result_rows),
            output=result_rows,
        )

        recorded_results = (
            result_rows.groupby("incident_id", sort=False)
            .agg(
                recorded_results=(
                    "label",
                    lambda values: list(dict.fromkeys(values.dropna())),
                )
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(label))",
            len(result_rows),
            len(recorded_results),
            output=recorded_results,
        )

        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        result_incidents = recorded_results.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(recorded_results, incidents, incident_id)",
            {"left": len(recorded_results), "right": len(incidents)},
            len(result_incidents),
            output=result_incidents,
        )

        reports = load_selected_texts("asrs", result_incidents)[
            ["incident_id", "recorded_results", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=result_incidents.text_file)",
            len(result_incidents),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(maintenance is primary disruption driver)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                maintenance_driven = reports.copy()
            else:
                maintenance_driven = reports.sem_filter(
                    "Keep report {text} only when maintenance is the primary driver "
                    "of the recorded disruption, rather than weather, crew legality, "
                    "passenger issues, or another non-maintenance cause."
                )
            step.set_output(maintenance_driven)

        with tracker.step(
            "SEM_FILTER(specific affected component or item identifiable)",
            input_rows=len(maintenance_driven),
        ) as step:
            if maintenance_driven.empty:
                component_identified = maintenance_driven.copy()
            else:
                component_identified = maintenance_driven.sem_filter(
                    "Keep report {text} only when it identifies a specific affected "
                    "aircraft component or required maintenance item."
                )
            step.set_output(component_identified)

        with tracker.step(
            "SEM_EXTRACT(affected item and maintenance driver phrase)",
            input_rows=len(component_identified),
        ) as step:
            if component_identified.empty:
                maintenance_candidates = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "recorded_results",
                        "narrative_affected_item",
                        "maintenance_driver_phrase",
                        "text",
                    ]
                )
            else:
                extracted = component_identified.sem_extract(
                    input_cols=["recorded_results", "text"],
                    output_cols={
                        "narrative_affected_item": (
                            "the specific affected component or required item using "
                            "a concise phrase faithful to the narrative"
                        ),
                        "maintenance_driver_phrase": (
                            "a concise phrase stating the maintenance condition or "
                            "action that primarily drove the disruption"
                        ),
                    },
                )
                for column in (
                    "narrative_affected_item",
                    "maintenance_driver_phrase",
                ):
                    extracted[column] = extracted[column].map(
                        lambda value: clean_text(value, default="")
                    )
                complete = extracted[
                    ["narrative_affected_item", "maintenance_driver_phrase"]
                ].ne("").all(axis=1)
                maintenance_candidates = extracted.loc[
                    complete,
                    [
                        "incident_id",
                        "recorded_results",
                        "narrative_affected_item",
                        "maintenance_driver_phrase",
                        "text",
                    ],
                ].reset_index(drop=True)
            step.set_output(maintenance_candidates)

        components = load_table("asrs", "components.csv")[["attribute", "value"]].copy()
        tracker.record(
            "SCAN_TABLE(components.csv)",
            None,
            len(components),
            output=components,
        )

        vocabulary_rows = components[
            components["attribute"] == "aircraft_component"
        ].copy()
        tracker.record(
            "FILTER(attribute='aircraft_component')",
            len(components),
            len(vocabulary_rows),
            output=vocabulary_rows,
        )

        component_vocabulary = (
            vocabulary_rows[["value"]]
            .drop_duplicates()
            .rename(columns={"value": "canonical_component"})
            .reset_index(drop=True)
        )
        tracker.record(
            "DEDUP([value])",
            len(vocabulary_rows),
            len(component_vocabulary),
            output=component_vocabulary,
        )

        candidate_bindings = maintenance_candidates.copy()
        candidate_bindings["affected_item_binding"] = (
            "incident_id="
            + candidate_bindings["incident_id"].astype(str)
            + "; narrative_affected_item="
            + candidate_bindings["narrative_affected_item"].astype(str)
            + "; maintenance_driver_phrase="
            + candidate_bindings["maintenance_driver_phrase"].astype(str)
            + "; recorded_results="
            + candidate_bindings["recorded_results"].astype(str)
        )
        tracker.record(
            "CODE_MAP(bind narrative affected-item inputs)",
            len(maintenance_candidates),
            len(candidate_bindings),
            output=candidate_bindings,
        )

        vocabulary_bindings = component_vocabulary.copy()
        vocabulary_bindings["canonical_component_binding"] = (
            "canonical_component="
            + vocabulary_bindings["canonical_component"].astype(str)
        )
        tracker.record(
            "CODE_MAP(bind canonical component inputs)",
            len(component_vocabulary),
            len(vocabulary_bindings),
            output=vocabulary_bindings,
        )

        with tracker.step(
            "SEM_JOIN(affected item to canonical component)",
            input_rows={
                "left": len(candidate_bindings),
                "right": len(vocabulary_bindings),
            },
        ) as step:
            if candidate_bindings.empty or vocabulary_bindings.empty:
                semantic_component_matches = pd.DataFrame(
                    columns=[*candidate_bindings.columns, *vocabulary_bindings.columns]
                )
            else:
                semantic_component_matches = candidate_bindings.sem_join(
                    vocabulary_bindings,
                    "Match narrative affected item {affected_item_binding} to "
                    "canonical component {canonical_component_binding} only when "
                    "they are synonyms, abbreviations, or a clear part-to-system "
                    "relationship. Incident identity and generic maintenance or "
                    "aircraft-topic similarity are not evidence of a match."
                )
            step.set_output(semantic_component_matches)

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
            aircraft_attributes["attribute"].isin(MAINTENANCE_STATUS_ATTRIBUTES)
        ].copy()
        tracker.record(
            "FILTER(attribute IN target maintenance status attributes)",
            len(aircraft_attributes),
            len(status_rows),
            output=status_rows,
        )

        status_records = []
        for incident_id, group in status_rows.groupby("incident_id", sort=False):
            status_records.append(
                {
                    "incident_id": incident_id,
                    "maintenance_status_records": [
                        {
                            "attribute": str(row.attribute),
                            "value": str(row.value),
                        }
                        for row in group.itertuples(index=False)
                    ],
                }
            )
        maintenance_status_records = pd.DataFrame.from_records(
            status_records,
            columns=["incident_id", "maintenance_status_records"],
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(attribute, value))",
            len(status_rows),
            len(maintenance_status_records),
            output=maintenance_status_records,
        )

        joined = semantic_component_matches.merge(
            maintenance_status_records,
            on="incident_id",
            how="left",
            validate="many_to_one",
        )
        joined["maintenance_status_records"] = joined[
            "maintenance_status_records"
        ].map(lambda value: value if isinstance(value, list) else [])
        tracker.record(
            "LEFT_JOIN(component_matches, maintenance_status_records, incident_id)",
            {
                "left": len(semantic_component_matches),
                "right": len(maintenance_status_records),
            },
            len(joined),
            output=joined,
        )

        with tracker.step(
            "SEM_EXTRACT(classify maintenance driver)",
            input_rows=len(joined),
        ) as step:
            if joined.empty:
                classified_drivers = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "canonical_component",
                        "recorded_results",
                        "maintenance_driver",
                    ]
                )
            else:
                classified = joined.sem_extract(
                    input_cols=[
                        "maintenance_driver_phrase",
                        "recorded_results",
                        "maintenance_status_records",
                        "text",
                    ],
                    output_cols={
                        "maintenance_driver": (
                            "classify the primary maintenance driver exactly as "
                            "component_discrepancy_or_failure, mel_or_deferred_item, "
                            "troubleshooting_or_repair, "
                            "documentation_or_release_to_service, or "
                            "required_inspection"
                        )
                    },
                )
                classified["maintenance_driver"] = classified[
                    "maintenance_driver"
                ].map(lambda value: normalize_enum(value, MAINTENANCE_DRIVERS))
                classified_drivers = classified.loc[
                    classified["maintenance_driver"].notna(),
                    [
                        "incident_id",
                        "canonical_component",
                        "recorded_results",
                        "maintenance_driver",
                    ],
                ].reset_index(drop=True)
            step.set_output(classified_drivers)

        if classified_drivers.empty:
            grouped = pd.DataFrame(
                columns=[
                    "maintenance_driver",
                    "canonical_component",
                    "incident_count",
                    "recorded_result_labels",
                ]
            )
        else:
            group_records = []
            for group_key, group in classified_drivers.groupby(
                ["maintenance_driver", "canonical_component"],
                sort=False,
            ):
                maintenance_driver, canonical_component = group_key
                result_labels = sorted(
                    {
                        label
                        for labels in group["recorded_results"]
                        for label in labels
                    }
                )
                group_records.append(
                    {
                        "maintenance_driver": maintenance_driver,
                        "canonical_component": canonical_component,
                        "incident_count": group["incident_id"].nunique(),
                        "recorded_result_labels": result_labels,
                    }
                )
            grouped = pd.DataFrame.from_records(group_records)
        tracker.record(
            "GROUP_BY([maintenance_driver, canonical_component], count and result set)",
            len(classified_drivers),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["incident_count", "maintenance_driver", "canonical_component"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([incident_count DESC, driver/component ASC])",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(15).copy()
        tracker.record("LIMIT(15)", len(ordered), len(limited), output=limited)

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and maintenance-driver component fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
