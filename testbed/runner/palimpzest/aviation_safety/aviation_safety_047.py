#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-047."""

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
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "aviation_safety-047"
CONTRIBUTING_LABELS = ("Aircraft", "Human Factors")
ATTRIBUTION_CATEGORIES = (
    "mainly_equipment_driven",
    "mainly_human_factor_driven",
    "mixed_equipment_human_factor",
)
PERSON_FACTOR_TYPES = (
    "human_factors",
    "communication_breakdown",
    "experience",
)


def has_both_labels(values) -> bool:
    labels = set(values)
    return all(label in labels for label in CONTRIBUTING_LABELS)


def unpack_summary(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(value).split("|||", maxsplit=1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Malformed semantic aggregate output: {value!r}")
    return parts[0], parts[1]


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        assessments = load_table(
            "asrs",
            "assessments.csv",
            ["incident_id", "assessment_type", "label"],
        )
        tracker.record("scan", None, assessments)

        contributing_rows = assessments.loc[
            (assessments["assessment_type"] == "contributing_factor")
            & assessments["label"].isin(CONTRIBUTING_LABELS)
        ].reset_index(drop=True)
        tracker.record("filter", len(assessments), contributing_rows)

        grouped_labels = (
            contributing_rows.groupby("incident_id", sort=False)
            .agg(contributing_labels=("label", list))
            .reset_index()
        )
        tracker.record("groupby", len(contributing_rows), grouped_labels)

        dual_factor_incidents = grouped_labels.loc[
            grouped_labels["contributing_labels"].map(has_both_labels)
        ].reset_index(drop=True)
        tracker.record("filter", len(grouped_labels), dual_factor_incidents)

        incidents = load_table(
            "asrs",
            "incidents.csv",
            ["incident_id", "text_file"],
        )
        tracker.record("scan", None, incidents)

        dual_factor_reports = dual_factor_incidents.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {"left": len(dual_factor_incidents), "right": len(incidents)},
            dual_factor_reports,
        )

        reports = load_selected_texts("asrs", dual_factor_reports)[
            ["incident_id", "contributing_labels", "text"]
        ]
        tracker.record("scan", len(dual_factor_reports), reports)

        extraction_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "recoverable_causal_chain",
                    "type": bool,
                    "desc": (
                        "True only when the report's causal chain can be recovered "
                        "clearly enough to extract all requested fields."
                    ),
                },
                {
                    "name": "initiating_event",
                    "type": str,
                    "desc": "The initiating event as a short phrase.",
                },
                {
                    "name": "human_factor_event",
                    "type": str,
                    "desc": "The human action or condition as a short phrase.",
                },
                {
                    "name": "intervention",
                    "type": str,
                    "desc": "The intervention as a short phrase.",
                },
                {
                    "name": "causal_order",
                    "type": str,
                    "desc": "The causal order as a short phrase.",
                },
            ],
            desc=(
                "Recover the causal chain and extract the initiating event, human "
                "action or condition, intervention, and causal order. Mark the "
                "chain unrecoverable when these relations are unclear."
            ),
            depends_on=["incident_id", "contributing_labels", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            reports,
            [
                "recoverable_causal_chain",
                "initiating_event",
                "human_factor_event",
                "intervention",
                "causal_order",
            ],
        )
        extracted["recoverable_causal_chain"] = extracted[
            "recoverable_causal_chain"
        ].map(parse_bool)
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            extraction_result,
            time.time() - started,
        )

        recovered = extracted.loc[
            extracted["recoverable_causal_chain"]
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), recovered)

        attribution_plan = memory_dataset(
            f"{TASK_ID}-attribution", recovered
        ).sem_map(
            cols=[
                {
                    "name": "clear_weighting",
                    "type": bool,
                    "desc": "True only when the causal weighting is clear.",
                },
                {
                    "name": "attribution_category",
                    "type": str,
                    "desc": (
                        "Exactly one of mainly_equipment_driven, "
                        "mainly_human_factor_driven, or "
                        "mixed_equipment_human_factor."
                    ),
                },
            ],
            desc=(
                "Use mainly_equipment_driven when equipment initiated the chain "
                "and human actions were secondary; mainly_human_factor_driven "
                "when a decision, procedure, communication, or awareness failure "
                "was primary despite equipment context; and "
                "mixed_equipment_human_factor when both were necessary. Mark "
                "unclear cases as not having clear weighting."
            ),
            depends_on=[
                "initiating_event",
                "human_factor_event",
                "intervention",
                "causal_order",
                "text",
            ],
        )
        started = time.time()
        attribution_result = attribution_plan.run(config)
        attributed_incidents = result_frame(
            attribution_result,
            recovered,
            ["clear_weighting", "attribution_category"],
        )
        attributed_incidents["clear_weighting"] = attributed_incidents[
            "clear_weighting"
        ].map(parse_bool)
        attributed_incidents["attribution_category"] = attributed_incidents[
            "attribution_category"
        ].map(lambda value: normalize_enum(value, ATTRIBUTION_CATEGORIES))
        tracker.record_semantic(
            "sem_map",
            len(recovered),
            attributed_incidents,
            attribution_result,
            time.time() - started,
        )

        attributed_incidents = attributed_incidents.loc[
            attributed_incidents["clear_weighting"]
            & attributed_incidents["attribution_category"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(attribution_result), attributed_incidents)

        components = load_table(
            "asrs",
            "components.csv",
            ["incident_id", "attribute", "value"],
        )
        tracker.record("scan", None, components)

        component_rows = components.loc[
            components["attribute"].isin(["aircraft_component", "problem"])
        ].copy()
        component_rows["component_record"] = list(
            zip(component_rows["attribute"], component_rows["value"])
        )
        component_rows = component_rows.reset_index(drop=True)
        tracker.record("filter", len(components), component_rows)

        component_records = (
            component_rows.groupby("incident_id", sort=False)
            .agg(component_records=("component_record", list))
            .reset_index()
        )
        tracker.record("groupby", len(component_rows), component_records)

        attributions_with_components = attributed_incidents.merge(
            component_records,
            on="incident_id",
            how="left",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(attributed_incidents),
                "right": len(component_records),
            },
            attributions_with_components,
        )

        person_factors = load_table(
            "asrs",
            "person_factors.csv",
            ["incident_id", "factor_type", "value"],
        )
        tracker.record("scan", None, person_factors)

        human_factor_rows = person_factors.loc[
            person_factors["factor_type"].isin(PERSON_FACTOR_TYPES)
        ].copy()
        human_factor_rows["human_factor_record"] = list(
            zip(human_factor_rows["factor_type"], human_factor_rows["value"])
        )
        human_factor_rows = human_factor_rows.reset_index(drop=True)
        tracker.record("filter", len(person_factors), human_factor_rows)

        human_factor_records = (
            human_factor_rows.groupby("incident_id", sort=False)
            .agg(human_factor_records=("human_factor_record", list))
            .reset_index()
        )
        tracker.record("groupby", len(human_factor_rows), human_factor_records)

        joined = attributions_with_components.merge(
            human_factor_records,
            on="incident_id",
            how="left",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(attributions_with_components),
                "right": len(human_factor_records),
            },
            joined,
        )

        joined = joined.copy()
        joined["causal_record"] = joined[
            [
                "initiating_event",
                "human_factor_event",
                "intervention",
                "component_records",
                "human_factor_records",
            ]
        ].to_dict(orient="records")
        grouped = (
            joined.groupby("attribution_category", sort=True)
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
                        "attribution_category": row["attribution_category"],
                        "causal_records": row["causal_records"],
                    }
                ]
            )
            aggregate_plan = memory_dataset(
                f"{TASK_ID}-aggregate-{row['attribution_category']}",
                aggregate_input,
            ).sem_agg(
                col={
                    "name": "attribution_summary",
                    "type": str,
                    "desc": (
                        "Exactly '<dominant initiating factor> ||| <dominant "
                        "intervention>', with each phrase no longer than six words."
                    ),
                },
                agg=(
                    "For this attribution category, summarize the dominant "
                    "initiating factor and intervention. Each phrase must contain "
                    "no more than six words. Return exactly the two phrases "
                    "separated by |||."
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
            initiating_factor, intervention = unpack_summary(
                aggregate_frame.iloc[0]["attribution_summary"]
            )
            summary_rows.append(
                {
                    "attribution_category": row["attribution_category"],
                    "incident_count": row["incident_count"],
                    "dominant_initiating_factor": initiating_factor,
                    "dominant_intervention": intervention,
                }
            )

        answer_frame = pd.DataFrame.from_records(
            summary_rows,
            columns=[
                "attribution_category",
                "incident_count",
                "dominant_initiating_factor",
                "dominant_intervention",
            ],
        )
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
