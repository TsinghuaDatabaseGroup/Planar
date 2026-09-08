#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-045."""

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

        connection_plan = memory_dataset(
            f"{TASK_ID}-maintenance-connection", reports
        ).sem_filter(
            filter=(
                "The report explicitly connects an earlier maintenance deferral "
                "or release-to-service decision to a later component discrepancy."
            ),
            depends_on=["text"],
        )
        started = time.time()
        connection_result = connection_plan.run(config)
        connected_reports = result_frame(connection_result, reports)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            connected_reports,
            connection_result,
            time.time() - started,
        )

        discovery_plan = memory_dataset(
            f"{TASK_ID}-discovery-phase", connected_reports
        ).sem_filter(
            filter=(
                "The report clearly places discovery before departure, after "
                "takeoff, or after landing."
            ),
            depends_on=["text"],
        )
        started = time.time()
        discovery_result = discovery_plan.run(config)
        located_reports = result_frame(
            discovery_result,
            connected_reports,
        )
        tracker.record_semantic(
            "sem_filter",
            len(connected_reports),
            located_reports,
            discovery_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-chain-details", located_reports
        ).sem_map(
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
                "Extract the earlier maintenance decision, affected item, later "
                "discrepancy, operational effect, and discovery phase."
            ),
            depends_on=["incident_id", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            located_reports,
            [
                "earlier_decision",
                "affected_item",
                "later_discrepancy",
                "operational_effect",
                "discovery_phase",
            ],
        )
        extracted["discovery_phase"] = extracted["discovery_phase"].map(
            lambda value: normalize_enum(value, DISCOVERY_PHASES)
        )
        tracker.record_semantic(
            "sem_map",
            len(located_reports),
            extracted,
            extraction_result,
            time.time() - started,
        )

        extracted = extracted.loc[
            extracted["discovery_phase"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extraction_result), extracted)

        category_plan = memory_dataset(
            f"{TASK_ID}-chain-category", extracted
        ).sem_map(
            cols=[
                {
                    "name": "chain_category",
                    "type": str,
                    "desc": (
                        "Exactly one of deferred_item_recurred, "
                        "released_item_failed, incomplete_repair_or_signoff, or "
                        "maintenance_limitation_escalated."
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
        started = time.time()
        category_result = category_plan.run(config)
        semantic_chains = result_frame(
            category_result,
            extracted,
            ["chain_category"],
        )
        semantic_chains["chain_category"] = semantic_chains[
            "chain_category"
        ].map(lambda value: normalize_enum(value, CHAIN_CATEGORIES))
        tracker.record_semantic(
            "sem_map",
            len(extracted),
            semantic_chains,
            category_result,
            time.time() - started,
        )

        semantic_chains = semantic_chains.loc[
            semantic_chains["chain_category"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(category_result), semantic_chains)

        aircraft_attributes = load_table(
            "asrs",
            "aircraft_attributes.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, aircraft_attributes)

        maintenance_rows = aircraft_attributes.loc[
            aircraft_attributes["attribute"].isin([DEFERRED_FLAG, RELEASED_FLAG])
            & (aircraft_attributes["value"] == "Y")
        ].reset_index(drop=True)
        tracker.record("filter", len(aircraft_attributes), maintenance_rows)

        maintenance_flags = (
            maintenance_rows.groupby("incident_id", sort=False)
            .agg(maintenance_flags=("attribute", list))
            .reset_index()
        )
        tracker.record("groupby", len(maintenance_rows), maintenance_flags)

        maintenance_flags["maintenance_status"] = maintenance_flags[
            "maintenance_flags"
        ].map(maintenance_status)
        maintenance_flags = maintenance_flags[
            ["incident_id", "maintenance_status"]
        ]
        tracker.record("project", len(maintenance_flags), maintenance_flags)

        chains_with_status = semantic_chains.merge(
            maintenance_flags,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {"left": len(semantic_chains), "right": len(maintenance_flags)},
            chains_with_status,
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

        joined = chains_with_status.merge(
            component_records,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {"left": len(chains_with_status), "right": len(component_records)},
            joined,
        )

        joined = joined.copy()
        joined["chain_record"] = joined[
            [
                "affected_item",
                "later_discrepancy",
                "structured_components",
                "structured_problems",
            ]
        ].to_dict(orient="records")
        group_keys = [
            "maintenance_status",
            "discovery_phase",
            "chain_category",
        ]
        grouped = (
            joined.groupby(group_keys, sort=True)
            .agg(
                incident_count=("incident_id", "nunique"),
                chain_records=("chain_record", list),
            )
            .reset_index()
        )
        tracker.record("groupby", len(joined), grouped)

        summary_rows = []
        for index, row in enumerate(grouped.to_dict(orient="records"), start=1):
            aggregate_input = pd.DataFrame(
                [
                    {
                        **{key: row[key] for key in group_keys},
                        "chain_records": row["chain_records"],
                    }
                ]
            )
            aggregate_plan = memory_dataset(
                f"{TASK_ID}-aggregate-{index}", aggregate_input
            ).sem_agg(
                col={
                    "name": "chain_summary",
                    "type": str,
                    "desc": (
                        "Exactly '<canonical affected item> ||| <recurring "
                        "discrepancy>', with each phrase no longer than six words."
                    ),
                },
                agg=(
                    "For this maintenance-status, discovery-phase, and chain-"
                    "category group, provide one canonical affected item and "
                    "recurring discrepancy. Each phrase must contain no more "
                    "than six words. Return exactly the two phrases separated by "
                    "|||."
                ),
                depends_on=["chain_records"],
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
            canonical_item, recurring_discrepancy = unpack_summary(
                aggregate_frame.iloc[0]["chain_summary"]
            )
            summary_rows.append(
                {
                    **{key: row[key] for key in group_keys},
                    "incident_count": row["incident_count"],
                    "canonical_affected_item": canonical_item,
                    "recurring_discrepancy": recurring_discrepancy,
                }
            )

        summaries = pd.DataFrame.from_records(
            summary_rows,
            columns=[
                *group_keys,
                "incident_count",
                "canonical_affected_item",
                "recurring_discrepancy",
            ],
        )
        ordered = summaries.sort_values(
            ["incident_count", *group_keys],
            ascending=[False, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("sort", len(summaries), ordered)

        limited = ordered.head(12).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited.copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
