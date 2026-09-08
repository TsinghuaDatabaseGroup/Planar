#!/usr/bin/env python3
"""Plan-optimization pipeline for aviation_safety-044."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    return any(term in str(value) for term in COMPONENT_TERMS)


def relevant_event(value) -> bool:
    text = str(value)
    return text == SMOKE_EVENT or "Aircraft Equipment Problem" in text


def normalize_cause(record: dict) -> dict:
    return {
        "normalized_cause_category": normalize_enum(
            record.get("cause_category"),
            CAUSE_CATEGORIES,
        )
    }


def has_valid_cause(record: dict) -> bool:
    return bool(record.get("normalized_cause_category"))


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

        components = load_table(
            "asrs",
            "components.csv",
            ["incident_id", "attribute", "value"],
        )
        component_rows = components.loc[
            (components["attribute"] == "aircraft_component")
            & components["value"].map(relevant_component)
        ].reset_index(drop=True)
        component_records = (
            component_rows.groupby("incident_id", sort=False)
            .agg(component_records=("value", list))
            .reset_index()
        )

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        event_rows = events.loc[events["label"].map(relevant_event)].copy()
        event_rows["event_record"] = list(
            zip(event_rows["event_type"], event_rows["label"])
        )
        event_records = (
            event_rows.groupby("incident_id", sort=False)
            .agg(event_labels=("event_record", list))
            .reset_index()
        )

        causal = memory_dataset(f"{TASK_ID}-reports", reports).sem_filter(
            (
                "Keep this report only if the narrative states one clearly "
                "primary cause for the smoke, fumes, odor, or equipment event."
            ),
            depends_on=["text"],
        )
        causal = causal.sem_filter(
            (
                "Keep this report only if it describes an operational response "
                "or consequence beyond a maintenance-record entry alone."
            ),
            depends_on=["text"],
        )
        causal = causal.sem_map(
            cols=[
                {
                    "name": "initiating_condition",
                    "type": str,
                    "desc": "The initiating condition in at most eight words.",
                },
                {
                    "name": "system_phrase",
                    "type": str,
                    "desc": "The relevant system in at most eight words.",
                },
                {
                    "name": "intervention",
                    "type": str,
                    "desc": "The crew or maintenance intervention in at most eight words.",
                },
                {
                    "name": "operational_outcome",
                    "type": str,
                    "desc": "The operational outcome in at most eight words.",
                },
            ],
            desc=(
                "Extract the initiating condition, relevant system, intervention, "
                "and operational outcome."
            ),
            depends_on=["incident_id", "text"],
        )
        causal = causal.sem_map(
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
        causal = causal.map(
            normalize_cause,
            cols=[
                {
                    "name": "normalized_cause_category",
                    "type": str | None,
                    "desc": "Validated primary cause category.",
                }
            ],
            depends_on=["cause_category"],
        )
        causal = causal.filter(
            has_valid_cause,
            depends_on=["normalized_cause_category"],
        )
        joined = causal.join(
            memory_dataset(f"{TASK_ID}-components", component_records),
            on="incident_id",
            how="inner",
        )
        joined = joined.join(
            memory_dataset(f"{TASK_ID}-events", event_records),
            on="incident_id",
            how="inner",
        )
        joined = joined.distinct(
            [
                "incident_id",
                "normalized_cause_category",
                "system_phrase",
                "operational_outcome",
            ]
        )
        grouped = joined.groupby(
            pz.GroupBySig(
                group_by_fields=["normalized_cause_category"],
                agg_funcs=["count", "list", "list", "list", "list"],
                agg_fields=[
                    "incident_id",
                    "system_phrase",
                    "operational_outcome",
                    "component_records",
                    "event_labels",
                ],
            )
        )
        plan = grouped.sem_map(
            cols=[
                {
                    "name": "canonical_system",
                    "type": str,
                    "desc": "Canonical system phrase of at most six words.",
                },
                {
                    "name": "dominant_operational_outcome",
                    "type": str,
                    "desc": "Dominant operational outcome in at most six words.",
                },
            ],
            desc=(
                "For this cause category, summarize the canonical system and "
                "dominant operational outcome from the collected records."
            ),
            depends_on=[
                "normalized_cause_category",
                "list(system_phrase)",
                "list(operational_outcome)",
                "list(component_records)",
                "list(event_labels)",
            ],
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True).rename(
            columns={
                "normalized_cause_category": "cause_category",
                "count(incident_id)": "incident_count",
            }
        )
        tracker.record_semantic(
            "optimized_plan",
            {
                "reports": len(reports),
                "components": len(component_records),
                "events": len(event_records),
            },
            output,
            optimized.result,
            time.time() - started,
        )
        answer = df_records(
            output[
                [
                    "cause_category",
                    "incident_count",
                    "canonical_system",
                    "dominant_operational_outcome",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
