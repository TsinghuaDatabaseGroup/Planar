#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-024."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    save_output,
    setup,
)


TASK_ID = "aviation_safety-024"


def main():
    setup(max_tokens=512)
    tracker = StepTracker()

    with Timer() as timer:
        left_incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "incident_date",
                "state_reference",
                "locale_reference_type",
                "primary_problem",
                "text_file",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv AS left_incident)",
            None,
            len(left_incidents),
            output=left_incidents,
        )
        left_reports = load_selected_texts("asrs", left_incidents)[
            [
                "incident_id",
                "incident_date",
                "state_reference",
                "locale_reference_type",
                "primary_problem",
                "text",
            ]
        ].copy()
        tracker.record(
            "SCAN_DOCS(selector=left_incident.text_file)",
            len(left_incidents),
            len(left_reports),
            output=left_reports,
        )

        right_incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "incident_date",
                "state_reference",
                "locale_reference_type",
                "primary_problem",
                "text_file",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv AS right_incident)",
            None,
            len(right_incidents),
            output=right_incidents,
        )
        right_reports = load_selected_texts("asrs", right_incidents)[
            [
                "incident_id",
                "incident_date",
                "state_reference",
                "locale_reference_type",
                "primary_problem",
                "text",
            ]
        ].copy()
        tracker.record(
            "SCAN_DOCS(selector=right_incident.text_file)",
            len(right_incidents),
            len(right_reports),
            output=right_reports,
        )

        left_reports["left_record"] = (
            "incident_id: "
            + left_reports["incident_id"].astype(str)
            + "\nincident_month: "
            + left_reports["incident_date"].astype(str)
            + "\nstate_reference: "
            + left_reports["state_reference"].fillna("").astype(str)
            + "\nlocale_reference_type: "
            + left_reports["locale_reference_type"].fillna("").astype(str)
            + "\nprimary_problem: "
            + left_reports["primary_problem"].fillna("").astype(str)
            + "\nnarrative: "
            + left_reports["text"].fillna("").astype(str)
        )
        left_bindings = left_reports[
            ["incident_id", "incident_date", "left_record"]
        ].rename(
            columns={
                "incident_id": "left_incident_id",
                "incident_date": "left_incident_date",
            }
        )
        tracker.record(
            "PROJECT(left semantic binding)",
            len(left_reports),
            len(left_bindings),
            output=left_bindings,
        )

        right_reports["right_record"] = (
            "incident_id: "
            + right_reports["incident_id"].astype(str)
            + "\nincident_month: "
            + right_reports["incident_date"].astype(str)
            + "\nstate_reference: "
            + right_reports["state_reference"].fillna("").astype(str)
            + "\nlocale_reference_type: "
            + right_reports["locale_reference_type"].fillna("").astype(str)
            + "\nprimary_problem: "
            + right_reports["primary_problem"].fillna("").astype(str)
            + "\nnarrative: "
            + right_reports["text"].fillna("").astype(str)
        )
        right_bindings = right_reports[
            ["incident_id", "incident_date", "right_record"]
        ].rename(
            columns={
                "incident_id": "right_incident_id",
                "incident_date": "right_incident_date",
            }
        )
        tracker.record(
            "PROJECT(right semantic binding)",
            len(right_reports),
            len(right_bindings),
            output=right_bindings,
        )

        with tracker.step(
            "SEM_JOIN(same underlying ASRS occurrence)",
            input_rows={
                "left": len(left_bindings),
                "right": len(right_bindings),
            },
        ) as step:
            matched_pairs = left_bindings.sem_join(
                right_bindings,
                "Evaluate left report {left_record:left} and right report "
                "{right_record:right} only when the left incident_id is smaller "
                "than the right incident_id and incident month, state reference, "
                "locale-reference type, and primary problem are equal. Keep the "
                "pair only when both narratives describe the same underlying "
                "occurrence, requiring a compatible distinctive event "
                "progression, measurements, actions, and outcome; generic topical "
                "similarity is insufficient."
            )
            step.set_output(matched_pairs)

        projected = matched_pairs[
            ["left_incident_date", "left_incident_id", "right_incident_id"]
        ].rename(columns={"left_incident_date": "incident_date"})
        projected = projected.drop_duplicates(
            subset=["left_incident_id", "right_incident_id"],
            ignore_index=True,
        )
        tracker.record(
            "PROJECT(incident_date, left_incident_id, right_incident_id)",
            len(matched_pairs),
            len(projected),
            output=projected,
        )

        ordered = projected.sort_values(
            ["incident_date", "left_incident_id", "right_incident_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(incident_date DESC, left_id ASC, right_id ASC)",
            len(projected),
            len(ordered),
            output=ordered,
        )
        ordered.insert(0, "rank", range(1, len(ordered) + 1))
        tracker.record(
            "PROJECT(rank=ROW_NUMBER(), result columns)",
            len(ordered),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
