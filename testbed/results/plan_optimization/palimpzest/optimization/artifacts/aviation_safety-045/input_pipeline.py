#!/usr/bin/env python3
"""Plan-optimization pipeline for aviation_safety-045."""

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
)

TASK_ID = "aviation_safety-045"
DISCOVERY_PHASES = ("before_departure", "after_takeoff", "after_landing")
CHAIN_CATEGORIES = (
    "deferred_item_recurred",
    "released_item_failed",
    "incomplete_repair_or_signoff",
    "maintenance_limitation_escalated",
)
DEFERRED_FLAG = "maintenance_status_maintenance_deferred"
RELEASED_FLAG = "maintenance_status_released_for_service"


def maintenance_status(values) -> str:
    flags = set(values)
    if DEFERRED_FLAG in flags and RELEASED_FLAG in flags:
        return "both"
    if DEFERRED_FLAG in flags:
        return "deferred"
    return "released_to_service"


def collect_for_attribute(
    values: pd.Series,
    frame: pd.DataFrame,
    attribute: str,
) -> list:
    return values.loc[
        frame.loc[values.index, "attribute"] == attribute
    ].dropna().tolist()


def normalize_chain(record: dict) -> dict:
    return {
        "normalized_discovery_phase": normalize_enum(
            record.get("discovery_phase"),
            DISCOVERY_PHASES,
        ),
        "normalized_chain_category": normalize_enum(
            record.get("chain_category"),
            CHAIN_CATEGORIES,
        ),
    }


def has_valid_chain(record: dict) -> bool:
    return bool(record.get("normalized_discovery_phase")) and bool(
        record.get("normalized_chain_category")
    )


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file", "primary_problem"],
        )
        aircraft_incidents = incidents.loc[
            incidents["primary_problem"] == "Aircraft"
        ].reset_index(drop=True)
        reports = load_selected_texts("asrs", aircraft_incidents)[
            ["incident_id", "text"]
        ]

        aircraft_attributes = load_table(
            "asrs",
            "aircraft_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        maintenance_rows = aircraft_attributes.loc[
            aircraft_attributes["attribute"].isin(
                [DEFERRED_FLAG, RELEASED_FLAG]
            )
            & (aircraft_attributes["value"] == "Y")
        ].reset_index(drop=True)
        maintenance_flags = (
            maintenance_rows.groupby("incident_id", sort=False)
            .agg(maintenance_flags=("attribute", list))
            .reset_index()
        )
        maintenance_flags["maintenance_status"] = maintenance_flags[
            "maintenance_flags"
        ].map(maintenance_status)
        maintenance_flags = maintenance_flags[
            ["incident_id", "maintenance_status"]
        ]

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

        chains = memory_dataset(f"{TASK_ID}-reports", reports).sem_filter(
            (
                "Keep this report only if it explicitly connects an earlier "
                "maintenance deferral or release-to-service decision to a "
                "later component discrepancy."
            ),
            depends_on=["text"],
        )
        chains = chains.sem_filter(
            (
                "Keep this report only if it clearly places discovery in "
                "before departure, after takeoff, or after landing."
            ),
            depends_on=["text"],
        )
        chains = chains.sem_map(
            cols=[
                {
                    "name": "earlier_decision",
                    "type": str,
                    "desc": "The earlier maintenance decision as a short phrase.",
                },
                {
                    "name": "affected_item",
                    "type": str,
                    "desc": "The affected item as a short phrase.",
                },
                {
                    "name": "later_discrepancy",
                    "type": str,
                    "desc": "The later discrepancy as a short phrase.",
                },
                {
                    "name": "operational_effect",
                    "type": str,
                    "desc": "The operational effect as a short phrase.",
                },
                {
                    "name": "discovery_phase",
                    "type": str,
                    "desc": (
                        "Exactly one of before_departure, after_takeoff, or "
                        "after_landing."
                    ),
                },
            ],
            desc=(
                "Extract the earlier maintenance decision, affected item, "
                "later discrepancy, operational effect, and discovery phase."
            ),
            depends_on=["incident_id", "text"],
        )
        chains = chains.sem_map(
            cols=[
                {
                    "name": "chain_category",
                    "type": str,
                    "desc": (
                        "Exactly one of deferred_item_recurred, "
                        "released_item_failed, incomplete_repair_or_signoff, "
                        "or maintenance_limitation_escalated."
                    ),
                }
            ],
            desc="Assign the causal maintenance-chain category.",
            depends_on=[
                "earlier_decision",
                "affected_item",
                "later_discrepancy",
                "operational_effect",
                "text",
            ],
        )
        chains = chains.map(
            normalize_chain,
            cols=[
                {
                    "name": "normalized_discovery_phase",
                    "type": str | None,
                    "desc": "Validated discovery-phase label.",
                },
                {
                    "name": "normalized_chain_category",
                    "type": str | None,
                    "desc": "Validated maintenance-chain label.",
                },
            ],
            depends_on=["discovery_phase", "chain_category"],
        ).filter(
            has_valid_chain,
            depends_on=[
                "normalized_discovery_phase",
                "normalized_chain_category",
            ],
        )
        joined = chains.join(
            memory_dataset(f"{TASK_ID}-maintenance", maintenance_flags),
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
                "maintenance_status",
                "normalized_discovery_phase",
                "normalized_chain_category",
                "affected_item",
                "later_discrepancy",
            ]
        )
        grouped = joined.groupby(
            pz.GroupBySig(
                group_by_fields=[
                    "maintenance_status",
                    "normalized_discovery_phase",
                    "normalized_chain_category",
                ],
                agg_funcs=["count", "list", "list", "list", "list"],
                agg_fields=[
                    "incident_id",
                    "affected_item",
                    "later_discrepancy",
                    "structured_components",
                    "structured_problems",
                ],
            )
        )
        plan = grouped.sem_map(
            cols=[
                {
                    "name": "canonical_affected_item",
                    "type": str,
                    "desc": "Canonical affected item in at most six words.",
                },
                {
                    "name": "recurring_discrepancy",
                    "type": str,
                    "desc": "Recurring discrepancy in at most six words.",
                },
            ],
            desc=(
                "For this maintenance-status, discovery-phase, and chain-"
                "category group, summarize one canonical affected item and "
                "one recurring discrepancy from the collected records."
            ),
            depends_on=[
                "maintenance_status",
                "normalized_discovery_phase",
                "normalized_chain_category",
                "list(affected_item)",
                "list(later_discrepancy)",
                "list(structured_components)",
                "list(structured_problems)",
            ],
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True).rename(
            columns={
                "normalized_discovery_phase": "discovery_phase",
                "normalized_chain_category": "chain_category",
                "count(incident_id)": "incident_count",
            }
        )
        tracker.record_semantic(
            "optimized_plan",
            {
                "reports": len(reports),
                "maintenance": len(maintenance_flags),
                "components": len(component_records),
            },
            output,
            optimized.result,
            time.time() - started,
        )
        ordered = output.sort_values(
            [
                "incident_count",
                "maintenance_status",
                "discovery_phase",
                "chain_category",
            ],
            ascending=[False, True, True, True],
            kind="stable",
        ).head(12).reset_index(drop=True)
        ordered.insert(0, "rank", range(1, len(ordered) + 1))
        answer = df_records(
            ordered[
                [
                    "rank",
                    "maintenance_status",
                    "discovery_phase",
                    "chain_category",
                    "incident_count",
                    "canonical_affected_item",
                    "recurring_discrepancy",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
