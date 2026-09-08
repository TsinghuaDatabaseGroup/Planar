#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-032."""

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
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-032"
ISSUE_CATEGORIES = (
    "passenger_medical",
    "security_or_behavior",
    "cabin_equipment_or_door",
    "boarding_or_deplaning",
    "flight_attendant_communication",
)
DECISION_CATEGORIES = (
    "delay_or_cancel",
    "return_or_divert",
    "continue_changed",
    "other_operational_change",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "far_part", "mission"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered = incidents[
            (incidents["far_part"] == "Part 121")
            & (incidents["mission"] == "Passenger")
        ].copy()
        tracker.record(
            "FILTER(far_part='Part 121' AND mission='Passenger')",
            len(incidents),
            len(filtered),
            output=filtered,
        )

        reports = load_selected_texts("asrs", filtered)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(cabin issue and changed flight-deck decision)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "qualifies": (
                        "true only when a cabin-side issue directly caused a "
                        "flight-deck operational change rather than affecting cabin "
                        "service only"
                    ),
                    "issue_category": (
                        "when qualifies is true, exactly one of passenger_medical, "
                        "security_or_behavior, cabin_equipment_or_door, "
                        "boarding_or_deplaning, flight_attendant_communication; "
                        "otherwise not_applicable"
                    ),
                    "decision_category": (
                        "when qualifies is true, exactly one of delay_or_cancel, "
                        "return_or_divert, continue_changed, "
                        "other_operational_change; otherwise not_applicable"
                    ),
                },
            )
            raw["qualifies"] = raw["qualifies"].map(parse_bool)
            raw["issue_category"] = raw["issue_category"].map(
                lambda value: normalize_enum(value, ISSUE_CATEGORIES)
            )
            raw["decision_category"] = raw["decision_category"].map(
                lambda value: normalize_enum(value, DECISION_CATEGORIES)
            )
            semantic_decisions = raw.loc[
                raw["qualifies"]
                & raw["issue_category"].notna()
                & raw["decision_category"].notna(),
                ["incident_id", "issue_category", "decision_category"],
            ].reset_index(drop=True)
            step.set_output(semantic_decisions)

        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        passenger_events = events[
            (events["event_type"] == "passengers_involved")
            & (events["label"] == "Y")
        ].copy()
        tracker.record(
            "FILTER(event_type='passengers_involved' AND label='Y')",
            len(events),
            len(passenger_events),
            output=passenger_events,
        )

        joined = semantic_decisions.merge(
            passenger_events[["incident_id"]],
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "JOIN(semantic_decisions, passenger_events, incident_id)",
            {
                "left": len(semantic_decisions),
                "right": len(passenger_events),
            },
            len(joined),
            output=joined,
        )

        grouped = (
            joined.groupby(
                ["issue_category", "decision_category"],
                sort=False,
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([issue_category, decision_category], COUNT_DISTINCT)",
            len(joined),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["incident_count", "issue_category", "decision_category"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([incident_count DESC, issue_category ASC, "
            "decision_category ASC])",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(10).copy()
        tracker.record(
            "LIMIT(10)",
            len(ordered),
            len(limited),
            output=limited,
        )

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank, issue_category, decision_category, incident_count)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
