#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-040."""

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
                "result_summary",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered_incidents = incidents[
            (incidents["far_part"] == "Part 121")
            & (incidents["locale_reference_type"] == "Airport")
        ].copy()
        tracker.record(
            "FILTER(far_part='Part 121' AND locale_reference_type='Airport')",
            len(incidents),
            len(filtered_incidents),
            output=filtered_incidents,
        )

        reports = load_selected_texts("asrs", filtered_incidents)[
            ["incident_id", "result_summary", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered_incidents),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(automation alert and first crew response)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                extracted_alerts = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "alert_or_advisory",
                        "first_crew_action",
                        "operational_context",
                        "highest_outcome",
                        "text",
                    ]
                )
            else:
                raw_alerts = reports.sem_extract(
                    input_cols=["result_summary", "text"],
                    output_cols={
                        "automation_record_supported": (
                            "true only when the report clearly identifies an "
                            "automation alert or aircraft advisory and the crew's "
                            "first response to it; otherwise false"
                        ),
                        "alert_or_advisory": (
                            "the specific automation alert or aircraft advisory in "
                            "at most six words"
                        ),
                        "first_crew_action": (
                            "the crew's first response to that alert or advisory as "
                            "a concise factual phrase"
                        ),
                        "operational_context": (
                            "the operational context in at most eight words"
                        ),
                        "highest_outcome": (
                            "the most consequential reported outcome, exactly one of "
                            "emergency_or_diversion, go_around_or_evasive_action, "
                            "continued_operation"
                        ),
                    },
                )
                raw_alerts["highest_outcome"] = raw_alerts[
                    "highest_outcome"
                ].map(lambda value: normalize_enum(value, OUTCOMES))
                supported = raw_alerts["automation_record_supported"].map(
                    parse_bool
                ) & raw_alerts["highest_outcome"].notna()
                extracted_alerts = raw_alerts.loc[
                    supported,
                    [
                        "incident_id",
                        "alert_or_advisory",
                        "first_crew_action",
                        "operational_context",
                        "highest_outcome",
                        "text",
                    ],
                ].copy()
                for column in (
                    "alert_or_advisory",
                    "first_crew_action",
                    "operational_context",
                ):
                    extracted_alerts[column] = extracted_alerts[column].map(
                        clean_text
                    )
                extracted_alerts = extracted_alerts.reset_index(drop=True)
            step.set_output(extracted_alerts)

        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        automation_rows = events[
            events["event_type"].isin(["detector", "result"])
            & events["label"].str.contains("Automation", regex=False, na=False)
        ].copy()
        tracker.record(
            "FILTER(event_type IN detector/result AND label CONTAINS Automation)",
            len(events),
            len(automation_rows),
            output=automation_rows,
        )

        automation_events = (
            automation_rows.groupby("incident_id", sort=False)
            .agg(
                automation_event_labels=(
                    "label",
                    lambda values: list(dict.fromkeys(values.dropna())),
                )
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(label))",
            len(automation_rows),
            len(automation_events),
            output=automation_events,
        )

        alert_candidates = extracted_alerts.merge(
            automation_events,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(extracted_alerts, automation_events, incident_id)",
            {"left": len(extracted_alerts), "right": len(automation_events)},
            len(alert_candidates),
            output=alert_candidates,
        )

        with tracker.step(
            "SEM_EXTRACT(classify first crew response)",
            input_rows=len(alert_candidates),
        ) as step:
            if alert_candidates.empty:
                response_records = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "alert_or_advisory",
                        "operational_context",
                        "response_label",
                        "highest_outcome",
                    ]
                )
            else:
                classified = alert_candidates.sem_extract(
                    input_cols=[
                        "first_crew_action",
                        "alert_or_advisory",
                        "text",
                    ],
                    output_cols={
                        "response_label": (
                            "classify the first crew action exactly as "
                            "follow_advisory, override_advisory, or "
                            "seek_clarification_before_acting"
                        )
                    },
                )
                classified["response_label"] = classified[
                    "response_label"
                ].map(lambda value: normalize_enum(value, RESPONSES))
                response_records = classified.loc[
                    classified["response_label"].notna(),
                    [
                        "incident_id",
                        "alert_or_advisory",
                        "operational_context",
                        "response_label",
                        "highest_outcome",
                    ],
                ].reset_index(drop=True)
            step.set_output(response_records)

        left_responses = response_records.rename(
            columns={column: f"left_{column}" for column in response_records.columns}
        ).copy()
        left_responses["left_response_binding"] = left_responses.apply(
            lambda row: (
                f"incident_id={row['left_incident_id']}; "
                f"alert_or_advisory={row['left_alert_or_advisory']}; "
                f"operational_context={row['left_operational_context']}; "
                f"response_label={row['left_response_label']}; "
                f"highest_outcome={row['left_highest_outcome']}"
            ),
            axis=1,
        )
        tracker.record(
            "PROJECT(bind left response records)",
            len(response_records),
            len(left_responses),
            output=left_responses,
        )

        right_responses = response_records.rename(
            columns={column: f"right_{column}" for column in response_records.columns}
        ).copy()
        right_responses["right_response_binding"] = right_responses.apply(
            lambda row: (
                f"incident_id={row['right_incident_id']}; "
                f"alert_or_advisory={row['right_alert_or_advisory']}; "
                f"operational_context={row['right_operational_context']}; "
                f"response_label={row['right_response_label']}; "
                f"highest_outcome={row['right_highest_outcome']}"
            ),
            axis=1,
        )
        tracker.record(
            "PROJECT(bind right response records)",
            len(response_records),
            len(right_responses),
            output=right_responses,
        )

        with tracker.step(
            "SEM_JOIN(equivalent alerts and contrasting first responses)",
            input_rows={"left": len(left_responses), "right": len(right_responses)},
        ) as step:
            if left_responses.empty or right_responses.empty:
                matched_pairs = pd.DataFrame(
                    columns=[*left_responses.columns, *right_responses.columns]
                )
            else:
                matched_pairs = left_responses.sem_join(
                    right_responses,
                    "Match left response {left_response_binding} with right "
                    "response {right_response_binding} only when they are different "
                    "incidents, their specific alerts or advisories and operational "
                    "contexts are semantically equivalent rather than merely sharing "
                    "generic automation vocabulary, and exactly one response is "
                    "follow_advisory while the other is override_advisory or "
                    "seek_clarification_before_acting."
                )
            step.set_output(matched_pairs)

        ordered_ids = matched_pairs[
            matched_pairs["left_incident_id"]
            < matched_pairs["right_incident_id"]
        ].copy()
        tracker.record(
            "FILTER(left.incident_id < right.incident_id)",
            len(matched_pairs),
            len(ordered_ids),
            output=ordered_ids,
        )

        projected = pd.DataFrame(
            {
                "left_incident_id": ordered_ids["left_incident_id"],
                "right_incident_id": ordered_ids["right_incident_id"],
                "alert_or_advisory": ordered_ids["left_alert_or_advisory"],
                "left_response": ordered_ids["left_response_label"],
                "right_response": ordered_ids["right_response_label"],
                "pair_priority": [
                    min(OUTCOME_PRIORITY[left], OUTCOME_PRIORITY[right])
                    for left, right in zip(
                        ordered_ids["left_highest_outcome"],
                        ordered_ids["right_highest_outcome"],
                        strict=True,
                    )
                ],
            }
        )
        tracker.record(
            "PROJECT(pair fields and pair_priority)",
            len(ordered_ids),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            ["pair_priority", "left_incident_id", "right_incident_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([pair_priority ASC, incident IDs ASC])",
            len(projected),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(10).copy()
        tracker.record("LIMIT(10)", len(ordered), len(limited), output=limited)

        result = limited.drop(columns=["pair_priority"]).copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and matched-pair fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
