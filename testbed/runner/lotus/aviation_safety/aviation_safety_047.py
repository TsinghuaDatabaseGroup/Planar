#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-047."""

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
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-047"
CONTRIBUTING_LABELS = {"Aircraft", "Human Factors"}
ATTRIBUTION_CATEGORIES = (
    "mainly_equipment_driven",
    "mainly_human_factor_driven",
    "mixed_equipment_human_factor",
)


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
        "dominant_initiating_factor": clean_text(
            parsed.get("dominant_initiating_factor"),
            default="",
        ),
        "dominant_intervention": clean_text(
            parsed.get("dominant_intervention"),
            default="",
        ),
    }


def normalize_group_key(value):
    return value[0] if isinstance(value, tuple) and len(value) == 1 else value


def collected_attribute_records(group: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"attribute": str(row.attribute), "value": str(row.value)}
        for row in group.itertuples(index=False)
    ]


def collected_factor_records(group: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"factor_type": str(row.factor_type), "value": str(row.value)}
        for row in group.itertuples(index=False)
    ]


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        assessments = load_table("asrs", "assessments.csv")[
            ["incident_id", "assessment_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(assessments.csv)",
            None,
            len(assessments),
            output=assessments,
        )

        contributing_rows = assessments[
            (assessments["assessment_type"] == "contributing_factor")
            & assessments["label"].isin(CONTRIBUTING_LABELS)
        ].copy()
        tracker.record(
            "FILTER(contributing_factor AND label IN Aircraft/Human Factors)",
            len(assessments),
            len(contributing_rows),
            output=contributing_rows,
        )

        grouped_labels = (
            contributing_rows.groupby("incident_id", sort=False)
            .agg(
                contributing_labels=(
                    "label",
                    lambda values: list(dict.fromkeys(values.dropna())),
                )
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(label))",
            len(contributing_rows),
            len(grouped_labels),
            output=grouped_labels,
        )

        dual_factor_incidents = grouped_labels[
            grouped_labels["contributing_labels"].map(
                lambda labels: CONTRIBUTING_LABELS.issubset(set(labels))
            )
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(labels CONTAIN Aircraft AND Human Factors)",
            len(grouped_labels),
            len(dual_factor_incidents),
            output=dual_factor_incidents,
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

        dual_factor_reports = dual_factor_incidents.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(dual_factor_incidents, incidents, incident_id)",
            {"left": len(dual_factor_incidents), "right": len(incidents)},
            len(dual_factor_reports),
            output=dual_factor_reports,
        )

        reports = load_selected_texts("asrs", dual_factor_reports)[
            ["incident_id", "contributing_labels", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=dual_factor_reports.text_file)",
            len(dual_factor_reports),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(equipment-human causal order)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                extracted_attributions = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "initiating_event",
                        "human_factor_event",
                        "intervention",
                        "causal_order",
                        "text",
                    ]
                )
            else:
                extracted = reports.sem_extract(
                    input_cols=["contributing_labels", "text"],
                    output_cols={
                        "causal_chain_supported": (
                            "true only when the narrative supports a recoverable "
                            "causal chain connecting equipment and human-factor "
                            "events; otherwise false"
                        ),
                        "initiating_event": (
                            "the event or condition that initiated the chain as a "
                            "short factual phrase"
                        ),
                        "human_factor_event": (
                            "the relevant human action or condition as a short phrase"
                        ),
                        "intervention": (
                            "the intervention in the causal chain as a short phrase"
                        ),
                        "causal_order": (
                            "the narrative causal ordering of equipment and human "
                            "events as a short phrase"
                        ),
                    },
                )
                for column in (
                    "initiating_event",
                    "human_factor_event",
                    "intervention",
                    "causal_order",
                ):
                    extracted[column] = extracted[column].map(
                        lambda value: clean_text(value, default="")
                    )
                complete = extracted[
                    [
                        "initiating_event",
                        "human_factor_event",
                        "intervention",
                        "causal_order",
                    ]
                ].ne("").all(axis=1)
                extracted_attributions = extracted.loc[
                    extracted["causal_chain_supported"].map(parse_bool) & complete,
                    [
                        "incident_id",
                        "initiating_event",
                        "human_factor_event",
                        "intervention",
                        "causal_order",
                        "text",
                    ],
                ].reset_index(drop=True)
            step.set_output(extracted_attributions)

        with tracker.step(
            "SEM_EXTRACT(classify equipment-human attribution)",
            input_rows=len(extracted_attributions),
        ) as step:
            if extracted_attributions.empty:
                attributed_incidents = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "initiating_event",
                        "human_factor_event",
                        "intervention",
                        "causal_order",
                        "attribution_category",
                    ]
                )
            else:
                classified = extracted_attributions.sem_extract(
                    input_cols=[
                        "initiating_event",
                        "human_factor_event",
                        "intervention",
                        "causal_order",
                        "text",
                    ],
                    output_cols={
                        "clear_attribution": (
                            "true only when the causal evidence clearly supports one "
                            "of the three attribution categories; otherwise false"
                        ),
                        "attribution_category": (
                            "mainly_equipment_driven when equipment initiated the "
                            "chain and human actions were secondary; "
                            "mainly_human_factor_driven when a decision, procedure, "
                            "communication, or awareness failure was primary despite "
                            "equipment context; mixed_equipment_human_factor when "
                            "both were necessary"
                        ),
                    },
                )
                classified["attribution_category"] = classified[
                    "attribution_category"
                ].map(
                    lambda value: normalize_enum(
                        value,
                        ATTRIBUTION_CATEGORIES,
                    )
                )
                attributed_incidents = classified.loc[
                    classified["clear_attribution"].map(parse_bool)
                    & classified["attribution_category"].notna(),
                    [
                        "incident_id",
                        "initiating_event",
                        "human_factor_event",
                        "intervention",
                        "causal_order",
                        "attribution_category",
                    ],
                ].reset_index(drop=True)
            step.set_output(attributed_incidents)

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

        component_record_rows = []
        for incident_id, group in component_rows.groupby("incident_id", sort=False):
            component_record_rows.append(
                {
                    "incident_id": incident_id,
                    "component_records": collected_attribute_records(group),
                }
            )
        component_records = pd.DataFrame.from_records(
            component_record_rows,
            columns=["incident_id", "component_records"],
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(attribute, value))",
            len(component_rows),
            len(component_records),
            output=component_records,
        )

        attributions_with_components = attributed_incidents.merge(
            component_records,
            on="incident_id",
            how="left",
            validate="one_to_one",
        )
        tracker.record(
            "LEFT_JOIN(attributed_incidents, component_records, incident_id)",
            {"left": len(attributed_incidents), "right": len(component_records)},
            len(attributions_with_components),
            output=attributions_with_components,
        )

        person_factors = load_table("asrs", "person_factors.csv")[
            ["incident_id", "factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv)",
            None,
            len(person_factors),
            output=person_factors,
        )

        human_factor_rows = person_factors[
            person_factors["factor_type"].isin(
                ["human_factors", "communication_breakdown", "experience"]
            )
        ].copy()
        tracker.record(
            "FILTER(factor_type IN human_factors/communication/experience)",
            len(person_factors),
            len(human_factor_rows),
            output=human_factor_rows,
        )

        human_factor_record_rows = []
        for incident_id, group in human_factor_rows.groupby(
            "incident_id",
            sort=False,
        ):
            human_factor_record_rows.append(
                {
                    "incident_id": incident_id,
                    "human_factor_records": collected_factor_records(group),
                }
            )
        human_factor_records = pd.DataFrame.from_records(
            human_factor_record_rows,
            columns=["incident_id", "human_factor_records"],
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(factor_type, value))",
            len(human_factor_rows),
            len(human_factor_records),
            output=human_factor_records,
        )

        joined = attributions_with_components.merge(
            human_factor_records,
            on="incident_id",
            how="left",
            validate="one_to_one",
        )
        tracker.record(
            "LEFT_JOIN(attributions_with_components, human_factor_records, incident_id)",
            {
                "left": len(attributions_with_components),
                "right": len(human_factor_records),
            },
            len(joined),
            output=joined,
        )

        causal_groups = []
        for category, group in joined.groupby("attribution_category", sort=False):
            causal_records = []
            for row in group.itertuples(index=False):
                causal_records.append(
                    {
                        "initiating_event": row.initiating_event,
                        "human_factor_event": row.human_factor_event,
                        "intervention": row.intervention,
                        "component_records": (
                            row.component_records
                            if isinstance(row.component_records, list)
                            else []
                        ),
                        "human_factor_records": (
                            row.human_factor_records
                            if isinstance(row.human_factor_records, list)
                            else []
                        ),
                    }
                )
            causal_groups.append(
                {
                    "attribution_category": category,
                    "incident_count": group["incident_id"].nunique(),
                    "causal_records": causal_records,
                }
            )
        grouped = pd.DataFrame.from_records(
            causal_groups,
            columns=["attribution_category", "incident_count", "causal_records"],
        )
        tracker.record(
            "GROUP_BY([attribution_category], count and causal records)",
            len(joined),
            len(grouped),
            output=grouped,
        )

        with tracker.step(
            "SEM_AGGREGATE(dominant initiating factor and intervention)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                aggregate_output = pd.DataFrame(
                    columns=["attribution_category", "_aggregate_output"]
                )
            else:
                aggregate_output = grouped[
                    ["attribution_category", "causal_records"]
                ].sem_agg(
                    "For attribution category {attribution_category}, use causal "
                    "records {causal_records} to return only a JSON object with "
                    "dominant_initiating_factor and dominant_intervention as string "
                    "fields, each no longer than six words.",
                    suffix="_aggregate_output",
                    group_by=["attribution_category"],
                )
                aggregate_output["attribution_category"] = aggregate_output[
                    "attribution_category"
                ].map(normalize_group_key)
            step.set_output(aggregate_output)

        aggregate_records = []
        for row in aggregate_output.to_dict(orient="records"):
            aggregate_records.append(
                {
                    "attribution_category": row["attribution_category"],
                    **parse_aggregate_json(row["_aggregate_output"]),
                }
            )
        parsed_aggregates = pd.DataFrame.from_records(
            aggregate_records,
            columns=[
                "attribution_category",
                "dominant_initiating_factor",
                "dominant_intervention",
            ],
        )

        result = grouped[
            ["attribution_category", "incident_count"]
        ].merge(
            parsed_aggregates,
            on="attribution_category",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "PROJECT(attribution count and dominant summaries)",
            len(grouped),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
