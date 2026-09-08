#!/usr/bin/env python3
"""Palimpzest pipeline for aviation_safety-040."""

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

TASK_ID = "aviation_safety-040"
OUTCOMES = (
    "emergency_or_diversion",
    "go_around_or_evasive_action",
    "continued_operation",
)
RESPONSES = (
    "follow_advisory",
    "override_advisory",
    "seek_clarification_before_acting",
)
OUTCOME_PRIORITY = {
    "emergency_or_diversion": 1,
    "go_around_or_evasive_action": 2,
    "continued_operation": 3,
}
JOIN_COLUMNS = [
    "incident_id",
    "alert_or_advisory",
    "operational_context",
    "response_label",
    "highest_outcome",
    "incident_id_right",
    "alert_or_advisory_right",
    "operational_context_right",
    "response_label_right",
    "highest_outcome_right",
]


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        incidents = load_table(
            "asrs",
            "incidents.csv",
            [
                "incident_id",
                "text_file",
                "far_part",
                "locale_reference_type",
                "result_summary",
            ],
        )
        tracker.record("scan", None, incidents)

        part121_airport = incidents.loc[
            (incidents["far_part"] == "Part 121")
            & (incidents["locale_reference_type"] == "Airport")
        ].reset_index(drop=True)
        tracker.record("filter", len(incidents), part121_airport)

        reports = load_selected_texts("asrs", part121_airport)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record("scan", len(part121_airport), reports)

        extraction_plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "qualifies",
                    "type": bool,
                    "desc": (
                        "True only when both an automation alert or aircraft "
                        "advisory and the crew's first response are clear."
                    ),
                },
                {
                    "name": "alert_or_advisory",
                    "type": str,
                    "desc": (
                        "The automation alert or aircraft advisory in no more "
                        "than six words."
                    ),
                },
                {
                    "name": "first_crew_action",
                    "type": str,
                    "desc": "The crew's first response to the alert or advisory.",
                },
                {
                    "name": "operational_context",
                    "type": str,
                    "desc": "The operational context in no more than eight words.",
                },
                {
                    "name": "highest_outcome",
                    "type": str,
                    "desc": (
                        "Exactly one of emergency_or_diversion, "
                        "go_around_or_evasive_action, or continued_operation."
                    ),
                },
            ],
            desc=(
                "Emit a qualifying classification only when an automation alert "
                "or aircraft advisory and the crew's first response are clear. "
                "Extract the requested alert, first response, operational context, "
                "and the most consequential reported outcome."
            ),
            depends_on=["incident_id", "result_summary", "text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            reports,
            [
                "qualifies",
                "alert_or_advisory",
                "first_crew_action",
                "operational_context",
                "highest_outcome",
            ],
        )
        extracted["qualifies"] = extracted["qualifies"].map(parse_bool)
        extracted["highest_outcome"] = extracted["highest_outcome"].map(
            lambda value: normalize_enum(value, OUTCOMES)
        )
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted,
            extraction_result,
            time.time() - started,
        )

        extracted_alerts = extracted.loc[
            extracted["qualifies"]
            & extracted["alert_or_advisory"].notna()
            & extracted["first_crew_action"].notna()
            & extracted["operational_context"].notna()
            & extracted["highest_outcome"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), extracted_alerts)

        events = load_table(
            "asrs",
            "events.csv",
            ["incident_id", "event_type", "label"],
        )
        tracker.record("scan", None, events)

        automation_rows = events.loc[
            events["event_type"].isin(["detector", "result"])
            & events["label"].str.contains("Automation", regex=False, na=False)
        ].reset_index(drop=True)
        tracker.record("filter", len(events), automation_rows)

        automation_events = (
            automation_rows.groupby("incident_id", sort=False)
            .agg(automation_event_labels=("label", list))
            .reset_index()
        )
        tracker.record("groupby", len(automation_rows), automation_events)

        alert_candidates = extracted_alerts.merge(
            automation_events,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "join",
            {
                "left": len(extracted_alerts),
                "right": len(automation_events),
            },
            alert_candidates,
        )

        response_plan = memory_dataset(
            f"{TASK_ID}-response", alert_candidates
        ).sem_map(
            cols=[
                {
                    "name": "response_label",
                    "type": str,
                    "desc": (
                        "Exactly one of follow_advisory, override_advisory, or "
                        "seek_clarification_before_acting."
                    ),
                }
            ],
            desc="Classify the crew's first action in response to the alert.",
            depends_on=[
                "first_crew_action",
                "alert_or_advisory",
                "text",
            ],
        )
        started = time.time()
        response_result = response_plan.run(config)
        classified = result_frame(
            response_result,
            alert_candidates,
            ["response_label"],
        )
        classified["response_label"] = classified["response_label"].map(
            lambda value: normalize_enum(value, RESPONSES)
        )
        tracker.record_semantic(
            "sem_map",
            len(alert_candidates),
            classified,
            response_result,
            time.time() - started,
        )

        response_records = classified.loc[
            classified["response_label"].notna()
        ].reset_index(drop=True)
        tracker.record("filter", len(classified), response_records)

        left_responses = memory_dataset(
            f"{TASK_ID}-left", response_records
        )
        right_responses = memory_dataset(
            f"{TASK_ID}-right", response_records
        )
        pair_plan = left_responses.sem_join(
            right_responses,
            condition=(
                "Match only two different incidents whose alerts or advisories "
                "and operational contexts are semantically equivalent, rather "
                "than merely sharing generic automation vocabulary. One response "
                "must be follow_advisory and the other must be either "
                "override_advisory or seek_clarification_before_acting."
            ),
            depends_on=JOIN_COLUMNS,
        )
        started = time.time()
        pair_result = pair_plan.run(config)
        matched = result_frame(pair_result)
        if matched.empty:
            matched = pd.DataFrame(columns=JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(response_records), "right": len(response_records)},
            matched,
            pair_result,
            time.time() - started,
        )

        matched_pairs = matched.loc[
            matched["incident_id"] < matched["incident_id_right"]
        ].reset_index(drop=True)
        tracker.record("filter", len(matched), matched_pairs)

        projected = pd.DataFrame(
            {
                "left_incident_id": matched_pairs["incident_id"],
                "right_incident_id": matched_pairs["incident_id_right"],
                "alert_or_advisory": matched_pairs["alert_or_advisory"],
                "left_response": matched_pairs["response_label"],
                "right_response": matched_pairs["response_label_right"],
            }
        )
        if matched_pairs.empty:
            projected["pair_priority"] = pd.Series(dtype="int64")
        else:
            projected["pair_priority"] = matched_pairs.apply(
                lambda row: min(
                    OUTCOME_PRIORITY[row["highest_outcome"]],
                    OUTCOME_PRIORITY[row["highest_outcome_right"]],
                ),
                axis=1,
            )
        tracker.record("project", len(matched_pairs), projected)

        ordered = projected.sort_values(
            ["pair_priority", "left_incident_id", "right_incident_id"],
            ascending=[True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("sort", len(projected), ordered)

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited.drop(columns=["pair_priority"]).copy()
        answer_frame.insert(0, "rank", range(1, len(answer_frame) + 1))
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
