#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-044."""

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

TASK_ID = "aviation_safety-044"
CAUSE_CATEGORIES = (
    "equipment_malfunction",
    "maintenance_or_release_decision",
    "environmental_contamination",
    "crew_procedure",
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
        "canonical_system": clean_text(
            parsed.get("canonical_system"),
            default="",
        ),
        "dominant_operational_outcome": clean_text(
            parsed.get("dominant_operational_outcome"),
            default="",
        ),
    }


def normalize_group_key(value):
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


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
            "SEM_FILTER(one clearly primary event cause)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                primary_cause_reports = reports.copy()
            else:
                primary_cause_reports = reports.sem_filter(
                    "Keep report {text} only when it states one clearly primary "
                    "cause for the smoke, fumes, odor, or aircraft equipment event."
                )
            step.set_output(primary_cause_reports)

        with tracker.step(
            "SEM_FILTER(operational consequence beyond maintenance entry)",
            input_rows=len(primary_cause_reports),
        ) as step:
            if primary_cause_reports.empty:
                operational_reports = primary_cause_reports.copy()
            else:
                operational_reports = primary_cause_reports.sem_filter(
                    "Keep report {text} only when it describes an operational "
                    "response or consequence beyond a maintenance-record entry "
                    "alone."
                )
            step.set_output(operational_reports)

        with tracker.step(
            "SEM_EXTRACT(causal equipment-event record)",
            input_rows=len(operational_reports),
        ) as step:
            if operational_reports.empty:
                extracted_causes = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "initiating_condition",
                        "system_phrase",
                        "intervention",
                        "operational_outcome",
                        "text",
                    ]
                )
            else:
                extracted = operational_reports.sem_extract(
                    input_cols=["text"],
                    output_cols={
                        "initiating_condition": (
                            "the condition that initiated the causal chain, in at "
                            "most eight words"
                        ),
                        "system_phrase": (
                            "the relevant aircraft system in at most eight words"
                        ),
                        "intervention": (
                            "the crew or maintenance intervention in at most eight "
                            "words"
                        ),
                        "operational_outcome": (
                            "the resulting operational outcome in at most eight words"
                        ),
                    },
                )
                for column in (
                    "initiating_condition",
                    "system_phrase",
                    "intervention",
                    "operational_outcome",
                ):
                    extracted[column] = extracted[column].map(
                        lambda value: clean_text(value, default="")
                    )
                complete = extracted[
                    [
                        "initiating_condition",
                        "system_phrase",
                        "intervention",
                        "operational_outcome",
                    ]
                ].ne("").all(axis=1)
                extracted_causes = extracted.loc[
                    complete,
                    [
                        "incident_id",
                        "initiating_condition",
                        "system_phrase",
                        "intervention",
                        "operational_outcome",
                        "text",
                    ],
                ].reset_index(drop=True)
            step.set_output(extracted_causes)

        with tracker.step(
            "SEM_EXTRACT(classify primary cause category)",
            input_rows=len(extracted_causes),
        ) as step:
            if extracted_causes.empty:
                causal_records = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "initiating_condition",
                        "system_phrase",
                        "intervention",
                        "operational_outcome",
                        "cause_category",
                    ]
                )
            else:
                classified = extracted_causes.sem_extract(
                    input_cols=[
                        "initiating_condition",
                        "system_phrase",
                        "intervention",
                        "operational_outcome",
                        "text",
                    ],
                    output_cols={
                        "cause_category": (
                            "classify the one primary cause exactly as "
                            "equipment_malfunction, "
                            "maintenance_or_release_decision, "
                            "environmental_contamination, or crew_procedure"
                        )
                    },
                )
                classified["cause_category"] = classified[
                    "cause_category"
                ].map(lambda value: normalize_enum(value, CAUSE_CATEGORIES))
                causal_records = classified.loc[
                    classified["cause_category"].notna(),
                    [
                        "incident_id",
                        "initiating_condition",
                        "system_phrase",
                        "intervention",
                        "operational_outcome",
                        "cause_category",
                    ],
                ].reset_index(drop=True)
            step.set_output(causal_records)

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
            (components["attribute"] == "aircraft_component")
            & components["value"].str.contains(
                "Pressurization|Pack|Bleed|Air Conditioning",
                regex=True,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(aircraft_component AND target pneumatic/environmental system)",
            len(components),
            len(component_rows),
            output=component_rows,
        )

        component_records = (
            component_rows.groupby("incident_id", sort=False)
            .agg(
                component_records=(
                    "value",
                    lambda values: list(dict.fromkeys(values.dropna())),
                )
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(component value))",
            len(component_rows),
            len(component_records),
            output=component_records,
        )

        causes_with_components = causal_records.merge(
            component_records,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(causal_records, component_records, incident_id)",
            {"left": len(causal_records), "right": len(component_records)},
            len(causes_with_components),
            output=causes_with_components,
        )

        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        event_rows = events[
            (events["label"] == "Flight Deck / Cabin / Aircraft Event Smoke / Fire / Fumes / Odor")
            | events["label"].str.contains(
                "Aircraft Equipment Problem",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(smoke/fumes/odor OR Aircraft Equipment Problem event)",
            len(events),
            len(event_rows),
            output=event_rows,
        )

        event_records = []
        for incident_id, group in event_rows.groupby("incident_id", sort=False):
            labels = list(
                {
                    (str(row.event_type), str(row.label))
                    for row in group.itertuples(index=False)
                }
            )
            event_records.append(
                {
                    "incident_id": incident_id,
                    "event_labels": [
                        {"event_type": event_type, "label": label}
                        for event_type, label in sorted(labels)
                    ],
                }
            )
        event_labels = pd.DataFrame.from_records(
            event_records,
            columns=["incident_id", "event_labels"],
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(event_type, label))",
            len(event_rows),
            len(event_labels),
            output=event_labels,
        )

        joined = causes_with_components.merge(
            event_labels,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(causes_with_components, event_labels, incident_id)",
            {"left": len(causes_with_components), "right": len(event_labels)},
            len(joined),
            output=joined,
        )

        grouped_records = []
        for cause_category, group in joined.groupby("cause_category", sort=False):
            grouped_records.append(
                {
                    "cause_category": cause_category,
                    "incident_count": group["incident_id"].nunique(),
                    "causal_records": [
                        {
                            "system_phrase": row.system_phrase,
                            "operational_outcome": row.operational_outcome,
                            "component_records": row.component_records,
                            "event_labels": row.event_labels,
                        }
                        for row in group.itertuples(index=False)
                    ],
                }
            )
        grouped = pd.DataFrame.from_records(
            grouped_records,
            columns=["cause_category", "incident_count", "causal_records"],
        )
        tracker.record(
            "GROUP_BY([cause_category], COUNT_DISTINCT and COLLECT records)",
            len(joined),
            len(grouped),
            output=grouped,
        )

        with tracker.step(
            "SEM_AGGREGATE(canonical system and dominant outcome by cause)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                aggregate_output = pd.DataFrame(
                    columns=["cause_category", "_aggregate_output"]
                )
            else:
                aggregate_output = grouped[
                    ["cause_category", "causal_records"]
                ].sem_agg(
                    "For cause category {cause_category}, use causal records "
                    "{causal_records} to return only a JSON object with two string "
                    "fields: canonical_system and dominant_operational_outcome. "
                    "Each value must be a factual phrase of at most six words.",
                    suffix="_aggregate_output",
                    group_by=["cause_category"],
                )
                aggregate_output["cause_category"] = aggregate_output[
                    "cause_category"
                ].map(normalize_group_key)
            step.set_output(aggregate_output)

        aggregate_records = []
        for row in aggregate_output.to_dict(orient="records"):
            aggregate_records.append(
                {
                    "cause_category": row["cause_category"],
                    **parse_aggregate_json(row["_aggregate_output"]),
                }
            )
        parsed_aggregates = pd.DataFrame.from_records(
            aggregate_records,
            columns=[
                "cause_category",
                "canonical_system",
                "dominant_operational_outcome",
            ],
        )

        result = grouped[
            ["cause_category", "incident_count"]
        ].merge(
            parsed_aggregates,
            on="cause_category",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "PROJECT(cause count, canonical system, dominant outcome)",
            len(grouped),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
