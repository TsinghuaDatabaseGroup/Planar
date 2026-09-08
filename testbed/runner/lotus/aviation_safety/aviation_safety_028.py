#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-028."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-028"
RESPONSES = (
    "followed_automation",
    "overrode_or_disconnected",
    "sought_atc_clarification",
)
FOLLOWED_LABEL = "Flight Crew FLC complied w / Automation / Advisory"
OVERRIDE_LABELS = {
    "Flight Crew FLC Overrode Automation",
    "Flight Crew Overrode Automation",
}


def structured_response(event_records) -> str | None:
    labels = {
        record.get("label")
        for record in event_records
        if isinstance(record, dict)
    }
    if FOLLOWED_LABEL in labels:
        return "followed_automation"
    if labels & OVERRIDE_LABELS:
        return "overrode_or_disconnected"
    return None


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "text_file",
                "far_part",
                "locale_reference_type",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        incident_candidates = incidents[
            (incidents["far_part"] == "Part 121")
            & (incidents["locale_reference_type"] == "Airport")
        ].copy()
        tracker.record(
            "FILTER(far_part='Part 121' AND locale_reference_type='Airport')",
            len(incidents),
            len(incident_candidates),
            output=incident_candidates,
        )

        reports = load_selected_texts("asrs", incident_candidates)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(incident_candidates),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(dominant narrative automation response)",
            input_rows=len(reports),
        ) as step:
            semantic_responses = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "narrative_response": (
                        "when the crew action is clear, assign exactly one dominant "
                        "response using this precedence: followed_automation, "
                        "overrode_or_disconnected, sought_atc_clarification. Return "
                        "null when no response is clear"
                    )
                },
            )
            semantic_responses["narrative_response"] = semantic_responses[
                "narrative_response"
            ].map(lambda value: normalize_enum(value, RESPONSES))
            semantic_responses = semantic_responses[
                ["incident_id", "narrative_response"]
            ].reset_index(drop=True)
            step.set_output(semantic_responses)

        aircraft_attributes = load_table("asrs", "aircraft_attributes.csv")[
            ["incident_id", "attribute", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(aircraft_attributes.csv)",
            None,
            len(aircraft_attributes),
            output=aircraft_attributes,
        )

        navigation = aircraft_attributes[
            (aircraft_attributes["attribute"] == "nav_in_use")
            & (
                aircraft_attributes["value"].str.contains(
                    "FMS Or FMC",
                    regex=False,
                    na=False,
                )
                | aircraft_attributes["value"].str.contains(
                    "GPS",
                    regex=False,
                    na=False,
                )
            )
        ].copy()
        tracker.record(
            "FILTER(attribute='nav_in_use' AND value CONTAINS FMS/FMC OR GPS)",
            len(aircraft_attributes),
            len(navigation),
            output=navigation,
        )

        navigation_records = (
            navigation.groupby("incident_id", sort=False)
            .agg(navigation_in_use=("value", list))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(navigation_in_use))",
            len(navigation),
            len(navigation_records),
            output=navigation_records,
        )

        responses_with_navigation = semantic_responses.merge(
            navigation_records,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(semantic_responses, navigation_records, incident_id)",
            {
                "left": len(semantic_responses),
                "right": len(navigation_records),
            },
            len(responses_with_navigation),
            output=responses_with_navigation,
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

        automation = events[
            events["event_type"].isin(["detector", "result"])
            & events["label"].str.contains(
                "Automation",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(event_type IN detector/result AND label CONTAINS Automation)",
            len(events),
            len(automation),
            output=automation,
        )

        automation["_event_record"] = automation[
            ["event_type", "label"]
        ].to_dict(orient="records")
        automation_events = (
            automation.groupby("incident_id", sort=False)
            .agg(automation_event_labels=("_event_record", list))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(event_type, label))",
            len(automation),
            len(automation_events),
            output=automation_events,
        )

        joined = responses_with_navigation.merge(
            automation_events,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(responses_with_navigation, automation_events, incident_id)",
            {
                "left": len(responses_with_navigation),
                "right": len(automation_events),
            },
            len(joined),
            output=joined,
        )

        projected = joined[["incident_id"]].copy()
        projected["response_pattern"] = [
            narrative
            if isinstance(narrative, str) and narrative in RESPONSES
            else structured_response(event_records)
            for narrative, event_records in zip(
                joined["narrative_response"],
                joined["automation_event_labels"],
                strict=True,
            )
        ]
        tracker.record(
            "PROJECT(response_pattern=COALESCE(narrative, structured CASE))",
            len(joined),
            len(projected),
            output=projected,
        )

        resolved = projected[projected["response_pattern"].notna()].copy()
        tracker.record(
            "FILTER(response_pattern IS NOT NULL)",
            len(projected),
            len(resolved),
            output=resolved,
        )

        grouped = (
            resolved.groupby("response_pattern", sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        response_order = {
            label: index for index, label in enumerate(RESPONSES)
        }
        grouped = grouped.sort_values(
            "response_pattern",
            key=lambda values: values.map(response_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([response_pattern], COUNT_DISTINCT)",
            len(resolved),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
