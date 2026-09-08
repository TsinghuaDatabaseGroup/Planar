#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-045."""

import json
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

TASK_ID = "aviation_safety-045"
DISCOVERY_PHASES = ("before_departure", "after_takeoff", "after_landing")
CHAIN_CATEGORIES = (
    "deferred_item_recurred",
    "released_item_failed",
    "incomplete_repair_or_signoff",
    "maintenance_limitation_escalated",
)
MAINTENANCE_FLAG_ATTRIBUTES = {
    "maintenance_status_maintenance_deferred",
    "maintenance_status_released_for_service",
}


def parse_aggregate_json(value) -> dict[str, str]:
    text = str(value).strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "canonical_affected_item": clean_text(
            parsed.get("canonical_affected_item"),
            default="",
        ),
        "recurring_discrepancy": clean_text(
            parsed.get("recurring_discrepancy"),
            default="",
        ),
    }


def normalized_group_value(value):
    return value[0] if isinstance(value, tuple) and len(value) == 1 else value


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
            "SEM_FILTER(maintenance decision caused later discrepancy)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                maintenance_chains = reports.copy()
            else:
                maintenance_chains = reports.sem_filter(
                    "Keep report {text} only when it explicitly connects an "
                    "earlier maintenance deferral or release-to-service decision "
                    "to a later component discrepancy."
                )
            step.set_output(maintenance_chains)

        with tracker.step(
            "SEM_FILTER(clear discovery phase)",
            input_rows=len(maintenance_chains),
        ) as step:
            if maintenance_chains.empty:
                phased_chains = maintenance_chains.copy()
            else:
                phased_chains = maintenance_chains.sem_filter(
                    "Keep report {text} only when it clearly places discovery of "
                    "the later discrepancy before departure, after takeoff, or "
                    "after landing."
                )
            step.set_output(phased_chains)

        with tracker.step(
            "SEM_EXTRACT(maintenance causal chain)",
            input_rows=len(phased_chains),
        ) as step:
            if phased_chains.empty:
                extracted_chains = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "earlier_decision",
                        "affected_item",
                        "later_discrepancy",
                        "operational_effect",
                        "discovery_phase",
                        "text",
                    ]
                )
            else:
                extracted = phased_chains.sem_extract(
                    input_cols=["text"],
                    output_cols={
                        "earlier_decision": (
                            "the earlier maintenance deferral or release decision "
                            "as a short factual phrase"
                        ),
                        "affected_item": (
                            "the affected aircraft item as a short factual phrase"
                        ),
                        "later_discrepancy": (
                            "the later component discrepancy as a short factual phrase"
                        ),
                        "operational_effect": (
                            "the operational effect as a short factual phrase"
                        ),
                        "discovery_phase": (
                            "exactly one of before_departure, after_takeoff, "
                            "after_landing"
                        ),
                    },
                )
                extracted["discovery_phase"] = extracted[
                    "discovery_phase"
                ].map(lambda value: normalize_enum(value, DISCOVERY_PHASES))
                for column in (
                    "earlier_decision",
                    "affected_item",
                    "later_discrepancy",
                    "operational_effect",
                ):
                    extracted[column] = extracted[column].map(
                        lambda value: clean_text(value, default="")
                    )
                complete = extracted[
                    [
                        "earlier_decision",
                        "affected_item",
                        "later_discrepancy",
                        "operational_effect",
                    ]
                ].ne("").all(axis=1)
                extracted_chains = extracted.loc[
                    complete & extracted["discovery_phase"].notna(),
                    [
                        "incident_id",
                        "earlier_decision",
                        "affected_item",
                        "later_discrepancy",
                        "operational_effect",
                        "discovery_phase",
                        "text",
                    ],
                ].reset_index(drop=True)
            step.set_output(extracted_chains)

        with tracker.step(
            "SEM_EXTRACT(classify maintenance chain category)",
            input_rows=len(extracted_chains),
        ) as step:
            if extracted_chains.empty:
                semantic_chains = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "affected_item",
                        "later_discrepancy",
                        "discovery_phase",
                        "chain_category",
                    ]
                )
            else:
                classified = extracted_chains.sem_extract(
                    input_cols=[
                        "earlier_decision",
                        "affected_item",
                        "later_discrepancy",
                        "operational_effect",
                        "text",
                    ],
                    output_cols={
                        "chain_category": (
                            "classify the causal chain exactly as "
                            "deferred_item_recurred, released_item_failed, "
                            "incomplete_repair_or_signoff, or "
                            "maintenance_limitation_escalated"
                        )
                    },
                )
                classified["chain_category"] = classified[
                    "chain_category"
                ].map(lambda value: normalize_enum(value, CHAIN_CATEGORIES))
                semantic_chains = classified.loc[
                    classified["chain_category"].notna(),
                    [
                        "incident_id",
                        "affected_item",
                        "later_discrepancy",
                        "discovery_phase",
                        "chain_category",
                    ],
                ].reset_index(drop=True)
            step.set_output(semantic_chains)

        aircraft_attributes = load_table("asrs", "aircraft_attributes.csv")[
            ["incident_id", "attribute", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(aircraft_attributes.csv)",
            None,
            len(aircraft_attributes),
            output=aircraft_attributes,
        )

        flag_rows = aircraft_attributes[
            aircraft_attributes["attribute"].isin(MAINTENANCE_FLAG_ATTRIBUTES)
            & (aircraft_attributes["value"] == "Y")
        ].copy()
        tracker.record(
            "FILTER(maintenance deferred/released flags with value='Y')",
            len(aircraft_attributes),
            len(flag_rows),
            output=flag_rows,
        )

        grouped_flags = (
            flag_rows.groupby("incident_id", sort=False)
            .agg(
                maintenance_flags=(
                    "attribute",
                    lambda values: list(dict.fromkeys(values.dropna())),
                )
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(attribute))",
            len(flag_rows),
            len(grouped_flags),
            output=grouped_flags,
        )

        maintenance_flags = grouped_flags[["incident_id"]].copy()
        maintenance_flags["maintenance_status"] = grouped_flags[
            "maintenance_flags"
        ].map(
            lambda flags: (
                "both"
                if MAINTENANCE_FLAG_ATTRIBUTES.issubset(set(flags))
                else (
                    "deferred"
                    if "maintenance_status_maintenance_deferred" in flags
                    else "released_to_service"
                )
            )
        )
        tracker.record(
            "PROJECT(maintenance_status=CASE flags)",
            len(grouped_flags),
            len(maintenance_flags),
            output=maintenance_flags,
        )

        chains_with_status = semantic_chains.merge(
            maintenance_flags,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(semantic_chains, maintenance_flags, incident_id)",
            {"left": len(semantic_chains), "right": len(maintenance_flags)},
            len(chains_with_status),
            output=chains_with_status,
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
            components["attribute"].isin(["aircraft_component", "problem"])
        ].copy()
        tracker.record(
            "FILTER(attribute IN aircraft_component/problem)",
            len(components),
            len(component_rows),
            output=component_rows,
        )

        component_records = []
        for incident_id, group in component_rows.groupby("incident_id", sort=False):
            component_records.append(
                {
                    "incident_id": incident_id,
                    "structured_components": group.loc[
                        group["attribute"] == "aircraft_component",
                        "value",
                    ].dropna().tolist(),
                    "structured_problems": group.loc[
                        group["attribute"] == "problem",
                        "value",
                    ].dropna().tolist(),
                }
            )
        structured_components = pd.DataFrame.from_records(
            component_records,
            columns=[
                "incident_id",
                "structured_components",
                "structured_problems",
            ],
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT_IF(component/problem))",
            len(component_rows),
            len(structured_components),
            output=structured_components,
        )

        joined = chains_with_status.merge(
            structured_components,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(chains_with_status, component_records, incident_id)",
            {"left": len(chains_with_status), "right": len(structured_components)},
            len(joined),
            output=joined,
        )

        group_columns = [
            "maintenance_status",
            "discovery_phase",
            "chain_category",
        ]
        group_records = []
        for group_key, group in joined.groupby(group_columns, sort=False):
            group_records.append(
                {
                    **dict(zip(group_columns, group_key, strict=True)),
                    "incident_count": group["incident_id"].nunique(),
                    "chain_records": [
                        {
                            "affected_item": row.affected_item,
                            "later_discrepancy": row.later_discrepancy,
                            "structured_components": row.structured_components,
                            "structured_problems": row.structured_problems,
                        }
                        for row in group.itertuples(index=False)
                    ],
                }
            )
        grouped = pd.DataFrame.from_records(
            group_records,
            columns=[
                *group_columns,
                "incident_count",
                "chain_records",
            ],
        )
        tracker.record(
            "GROUP_BY([maintenance_status, discovery_phase, chain_category])",
            len(joined),
            len(grouped),
            output=grouped,
        )

        with tracker.step(
            "SEM_AGGREGATE(canonical item and recurring discrepancy)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                aggregate_output = pd.DataFrame(
                    columns=[*group_columns, "_aggregate_output"]
                )
            else:
                aggregate_output = grouped[
                    [*group_columns, "chain_records"]
                ].sem_agg(
                    "For maintenance status {maintenance_status}, discovery phase "
                    "{discovery_phase}, and chain category {chain_category}, use "
                    "records {chain_records} to return only a JSON object with "
                    "canonical_affected_item and recurring_discrepancy as string "
                    "fields, each no longer than six words.",
                    suffix="_aggregate_output",
                    group_by=group_columns,
                )
                for column in group_columns:
                    aggregate_output[column] = aggregate_output[column].map(
                        normalized_group_value
                    )
            step.set_output(aggregate_output)

        aggregate_records = []
        for row in aggregate_output.to_dict(orient="records"):
            aggregate_records.append(
                {
                    **{column: row[column] for column in group_columns},
                    **parse_aggregate_json(row["_aggregate_output"]),
                }
            )
        parsed_aggregates = pd.DataFrame.from_records(
            aggregate_records,
            columns=[
                *group_columns,
                "canonical_affected_item",
                "recurring_discrepancy",
            ],
        )

        with_aggregates = grouped[
            [*group_columns, "incident_count"]
        ].merge(
            parsed_aggregates,
            on=group_columns,
            how="inner",
            validate="one_to_one",
        )
        ordered = with_aggregates.sort_values(
            ["incident_count", *group_columns],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([incident_count DESC, three labels ASC])",
            len(with_aggregates),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(12).copy()
        tracker.record("LIMIT(12)", len(ordered), len(limited), output=limited)

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and maintenance-chain fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
