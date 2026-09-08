#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-034."""

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
    parse_number_after,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-034"
PATTERNS = (
    "direct_conflict",
    "modified_or_delayed_compliance",
    "clarification_required",
)
AUTOMATION_DETECTORS = {"Automation Aircraft RA", "Automation Aircraft TA"}


def minimum_distance(labels, prefix):
    values = [
        parsed
        for label in labels
        if str(label).startswith(prefix)
        and (parsed := parse_number_after(label, prefix)) is not None
    ]
    return min(values) if values else None


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
            "SEM_EXTRACT(automation and ATC guidance reconciliation conflict)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "qualifies": (
                        "true only when automation guidance and an ATC instruction "
                        "were difficult to reconcile in the same episode"
                    ),
                    "reconciliation_pattern": (
                        "when qualifies is true, exactly one of direct_conflict, "
                        "modified_or_delayed_compliance, clarification_required; "
                        "otherwise not_applicable"
                    ),
                    "automation_guidance": (
                        "the relevant automation guidance as a natural phrase of at "
                        "most eight words; otherwise not_applicable"
                    ),
                    "atc_instruction": (
                        "the conflicting or difficult-to-reconcile ATC instruction "
                        "as a natural phrase of at most eight words; otherwise "
                        "not_applicable"
                    ),
                },
            )
            raw["qualifies"] = raw["qualifies"].map(parse_bool)
            raw["reconciliation_pattern"] = raw[
                "reconciliation_pattern"
            ].map(lambda value: normalize_enum(value, PATTERNS))
            raw["automation_guidance"] = raw["automation_guidance"].map(
                clean_text
            )
            raw["atc_instruction"] = raw["atc_instruction"].map(clean_text)
            semantic_conflicts = raw.loc[
                raw["qualifies"] & raw["reconciliation_pattern"].notna(),
                [
                    "incident_id",
                    "reconciliation_pattern",
                    "automation_guidance",
                    "atc_instruction",
                ],
            ].reset_index(drop=True)
            step.set_output(semantic_conflicts)

        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        relevant_events = events[
            (
                (events["event_type"] == "detector")
                & events["label"].isin(AUTOMATION_DETECTORS)
            )
            | (events["event_type"] == "miss_distance")
        ].copy()
        tracker.record(
            "FILTER(RA/TA detector OR event_type='miss_distance')",
            len(events),
            len(relevant_events),
            output=relevant_events,
        )

        event_rows = []
        for incident_id, group in relevant_events.groupby(
            "incident_id",
            sort=False,
        ):
            detector_labels = group.loc[
                group["event_type"] == "detector",
                "label",
            ].tolist()
            miss_labels = group.loc[
                group["event_type"] == "miss_distance",
                "label",
            ]
            event_rows.append(
                {
                    "incident_id": incident_id,
                    "automation_detectors": detector_labels,
                    "horizontal_separation_ft": minimum_distance(
                        miss_labels,
                        "Horizontal ",
                    ),
                    "vertical_separation_ft": minimum_distance(
                        miss_labels,
                        "Vertical ",
                    ),
                }
            )
        event_records = pd.DataFrame.from_records(
            event_rows,
            columns=[
                "incident_id",
                "automation_detectors",
                "horizontal_separation_ft",
                "vertical_separation_ft",
            ],
        )
        tracker.record(
            "GROUP_BY([incident_id], detectors and minimum miss distances)",
            len(relevant_events),
            len(event_records),
            output=event_records,
        )

        complete_events = event_records[
            event_records["automation_detectors"].map(bool)
            & event_records["horizontal_separation_ft"].notna()
            & event_records["vertical_separation_ft"].notna()
        ].copy()
        complete_events["horizontal_separation_ft"] = complete_events[
            "horizontal_separation_ft"
        ].astype(int)
        complete_events["vertical_separation_ft"] = complete_events[
            "vertical_separation_ft"
        ].astype(int)
        tracker.record(
            "FILTER(detector present and both miss distances present)",
            len(event_records),
            len(complete_events),
            output=complete_events,
        )

        joined = semantic_conflicts.merge(
            complete_events,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(semantic_conflicts, event_records, incident_id)",
            {
                "left": len(semantic_conflicts),
                "right": len(complete_events),
            },
            len(joined),
            output=joined,
        )

        severity_order = {label: index + 1 for index, label in enumerate(PATTERNS)}
        projected = joined[
            [
                "incident_id",
                "reconciliation_pattern",
                "automation_guidance",
                "atc_instruction",
                "horizontal_separation_ft",
                "vertical_separation_ft",
            ]
        ].copy()
        projected["severity_order"] = projected[
            "reconciliation_pattern"
        ].map(severity_order)
        tracker.record(
            "PROJECT(severity_order=CASE reconciliation_pattern)",
            len(joined),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            [
                "severity_order",
                "horizontal_separation_ft",
                "vertical_separation_ft",
                "incident_id",
            ],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([severity, horizontal, vertical, incident_id ASC])",
            len(projected),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(5).copy()
        tracker.record(
            "LIMIT(5)",
            len(ordered),
            len(limited),
            output=limited,
        )

        result = limited.drop(columns=["severity_order"]).copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and conflict fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
