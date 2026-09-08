#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-044."""

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

TASK_ID = "aviation_safety-044"
CAUSE_CATEGORIES = (
    "equipment_malfunction",
    "maintenance_or_release_decision",
    "environmental_contamination",
    "crew_procedure",
)
COMPONENT_TERMS = ("Pressurization", "Pack", "Bleed", "Air Conditioning")
SMOKE_EVENT = "Flight Deck / Cabin / Aircraft Event Smoke / Fire / Fumes / Odor"


def relevant_component(value) -> bool:
    text = str(value)
    return any(term in text for term in COMPONENT_TERMS)


def relevant_event(value) -> bool:
    text = str(value)
    return text == SMOKE_EVENT or "Aircraft Equipment Problem" in text


def unpack_summary(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(value).split("|||", maxsplit=1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Malformed semantic aggregate output: {value!r}")
    return parts[0], parts[1]


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

        aircraft_incidents = incidents.loc[
            incidents["primary_problem"] == "Aircraft"
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), aircraft_incidents)

        reports = load_selected_texts("asrs", aircraft_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record("scan", len(aircraft_incidents), reports)

        primary_cause_plan = memory_dataset(
            f"{TASK_ID}-primary-cause", reports
        ).sem_filter(
            filter=(
                "The narrative states one clearly primary cause for the smoke, "
                "fumes, odor, or aircraft-equipment event."
            ),
            depends_on=["text"],
        )
        started = time.time()
        primary_cause_result = primary_cause_plan.run(config)
        primary_cause_reports = result_frame(primary_cause_result, reports)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            primary_cause_reports,
            primary_cause_result,
            time.time() - started,
        )

        consequence_plan = memory_dataset(
            f"{TASK_ID}-operational-consequence", primary_cause_reports
        ).sem_filter(
            filter=(
                "The report describes an operational response or consequence "
                "beyond a maintenance-record entry alone."
            ),
            depends_on=["text"],
        )
        started = time.time()
        consequence_result = consequence_plan.run(config)
        consequence_reports = result_frame(
            consequence_result,
            primary_cause_reports,
        )
        tracker.record_semantic(
            "sem_filter",
            len(primary_cause_reports),
            consequence_reports,
            consequence_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-causal-details", consequence_reports
        ).sem_map(
            cols=[
                {
                    "name": "initiating_condition",
                    "type": str,
                    "desc": "The initiating condition in no more than eight words.",
                },
                {
                    "name": "system_phrase",
                    "type": str,
                    "desc": "The relevant system in no more than eight words.",
                },
                {
                    "name": "intervention",
                    "type": str,
                    "desc": (
                        "The crew or maintenance intervention in no more than "
                        "eight words."
                    ),
                },
                {
                    "name": "operational_outcome",
                    "type": str,
                    "desc": "The operational outcome in no more than eight words.",
                },
            ],
            desc=(
                "Extract the initiating condition, relevant system, crew or "
                "maintenance intervention, and operational outcome."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            consequence_reports,
            [
                "initiating_condition",
                "system_phrase",
                "intervention",
                "operational_outcome",
            ],
        )
        tracker.record_semantic(
            "sem_map",
            len(consequence_reports),
            extracted,
            extraction_result,
            time.time() - started,
        )

        classification_plan = memory_dataset(
            f"{TASK_ID}-cause-category", extracted
        ).sem_map(
            cols=[
                {
                    "name": "cause_category",
                    "type": str,
                    "desc": (
                        "Exactly one of equipment_malfunction, "
                        "maintenance_or_release_decision, "
                        "environmental_contamination, or crew_procedure."
                    ),
                }
            ],
            desc="Assign the report's one clearly primary cause category.",
            depends_on=[
                "initiating_condition",
                "system_phrase",
                "intervention",
                "operational_outcome",
                "text",
            ],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        causal_records = result_frame(
            classification_result,
            extracted,
            ["cause_category"],
        )
        causal_records["cause_category"] = causal_records["cause_category"].map(
            lambda value: normalize_enum(value, CAUSE_CATEGORIES)
        )
        tracker.record_semantic(
            "sem_map",
            len(extracted),
            causal_records,
            classification_result,
            time.time() - started,
        )

        causal_records = causal_records.loc[
            causal_records["cause_category"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(classification_result), causal_records)

        components = load_table(
            "asrs",
            "components.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, components)

        component_rows = components.loc[
            (components["attribute"] == "aircraft_component")
            & components["value"].map(relevant_component)
        ].reset_index(drop=True)
        tracker.record("filter", len(components), component_rows)

        component_records = (
            component_rows.groupby("incident_id", sort=False)
            .agg(component_records=("value", list))
            .reset_index()
        )
        tracker.record("groupby", len(component_rows), component_records)

        causes_with_components = causal_records.merge(
            component_records,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {"left": len(causal_records), "right": len(component_records)},
            causes_with_components,
        )

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        event_rows = events.loc[events["label"].map(relevant_event)].copy()
        event_rows["event_record"] = list(
            zip(event_rows["event_type"], event_rows["label"])
        )
        event_rows = event_rows.reset_index(drop=True)
        tracker.record("filter", len(events), event_rows)

        event_labels = (
            event_rows.groupby("incident_id", sort=False)
            .agg(event_labels=("event_record", list))
            .reset_index()
        )
        tracker.record("groupby", len(event_rows), event_labels)

        joined = causes_with_components.merge(
            event_labels,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(causes_with_components),
                "right": len(event_labels),
            },
            joined,
        )

        joined = joined.copy()
        joined["causal_record"] = joined[
            [
                "system_phrase",
                "operational_outcome",
                "component_records",
                "event_labels",
            ]
        ].to_dict(orient="records")
        grouped = (
            joined.groupby("cause_category", sort=True)
            .agg(
                incident_count=("incident_id", "nunique"),
                causal_records=("causal_record", list),
            )
            .reset_index()
        )
        tracker.record("groupby", len(joined), grouped)

        summary_rows = []
        for row in grouped.to_dict(orient="records"):
            aggregate_input = pd.DataFrame(
                [
                    {
                        "cause_category": row["cause_category"],
                        "causal_records": row["causal_records"],
                    }
                ]
            )
            aggregate_plan = memory_dataset(
                f"{TASK_ID}-aggregate-{row['cause_category']}", aggregate_input
            ).sem_agg(
                col={
                    "name": "cause_summary",
                    "type": str,
                    "desc": (
                        "Exactly '<canonical system> ||| <dominant operational "
                        "outcome>', with each phrase no longer than six words."
                    ),
                },
                agg=(
                    "For this cause category, summarize the canonical system and "
                    "dominant operational outcome. Each phrase must contain no "
                    "more than six words. Return exactly the two phrases separated "
                    "by |||."
                ),
                depends_on=["causal_records"],
            )
            started = time.time()
            aggregate_result = aggregate_plan.run(config)
            aggregate_frame = result_frame(aggregate_result)
            tracker.record_semantic(
                "sem_agg",
                len(aggregate_input),
                aggregate_frame,
                aggregate_result,
                time.time() - started,
            )
            canonical_system, dominant_outcome = unpack_summary(
                aggregate_frame.iloc[0]["cause_summary"]
            )
            summary_rows.append(
                {
                    "cause_category": row["cause_category"],
                    "incident_count": row["incident_count"],
                    "canonical_system": canonical_system,
                    "dominant_operational_outcome": dominant_outcome,
                }
            )

        answer_frame = pd.DataFrame.from_records(
            summary_rows,
            columns=[
                "cause_category",
                "incident_count",
                "canonical_system",
                "dominant_operational_outcome",
            ],
        )
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
