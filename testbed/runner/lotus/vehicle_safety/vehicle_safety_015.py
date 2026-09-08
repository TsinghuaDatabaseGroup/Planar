#!/usr/bin/env python3
"""LOTUS pipeline for vehicle_safety-015."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_jsonl,
    load_table,
    save_output,
    setup,
)


TASK_ID = "vehicle_safety-015"


def main():
    setup(max_tokens=384)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(complaints.csv)",
            None,
            len(complaints),
            output=complaints,
        )
        dodge_electrical_complaints = complaints[
            (complaints["make"] == "DODGE")
            & (complaints["component_id"] == "ELECTRICAL")
        ].copy()
        tracker.record(
            "FILTER(make='DODGE' AND component_id='ELECTRICAL')",
            len(complaints),
            len(dodge_electrical_complaints),
            output=dodge_electrical_complaints,
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")
        recalls["recall_row_id"] = recalls.index.astype(int)
        recalls = recalls[
            [
                "recall_row_id",
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_make": "make",
                "component_component_id": "component_id",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record(
            "SCAN_DOCS(recalls.jsonl)",
            None,
            len(recalls),
            output=recalls,
        )
        dodge_electrical_recalls = recalls[
            (recalls["make"] == "DODGE")
            & (recalls["component_id"] == "ELECTRICAL")
        ].copy()
        tracker.record(
            "FILTER(make='DODGE' AND component_id='ELECTRICAL')",
            len(recalls),
            len(dodge_electrical_recalls),
            output=dodge_electrical_recalls,
        )

        complaint_bindings = dodge_electrical_complaints[
            ["complaint_id", "summary_text"]
        ].rename(columns={"summary_text": "complaint_summary"})
        tracker.record(
            "PROJECT(complaint_id, complaint_summary)",
            len(dodge_electrical_complaints),
            len(complaint_bindings),
            output=complaint_bindings,
        )
        recall_bindings = dodge_electrical_recalls[
            ["recall_row_id", "campaign_number", "defect_summary"]
        ].copy()
        tracker.record(
            "PROJECT(recall_row_id, campaign_number, defect_summary)",
            len(dodge_electrical_recalls),
            len(recall_bindings),
            output=recall_bindings,
        )

        with tracker.step(
            "SEM_JOIN(same electrical failure with safety consequence)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            matched_pairs = complaint_bindings.sem_join(
                recall_bindings,
                "Keep complaint {complaint_summary:left} and recall defect "
                "{defect_summary:right} together only when both describe the "
                "same concrete electrical failure and the shared failure has a "
                "safety consequence such as stall, fire risk, loss of motive "
                "power, loss of lighting or warnings, unintended cruise "
                "behavior, or warning-system failure. Broad ELECTRICAL-category "
                "overlap is insufficient."
            )
            step.set_output(matched_pairs)

        distinct_pairs = matched_pairs[
            ["complaint_id", "recall_row_id", "campaign_number"]
        ].drop_duplicates(subset=["complaint_id", "recall_row_id"])
        grouped = (
            distinct_pairs.groupby("campaign_number", sort=False)
            .size()
            .rename("matched_pair_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(campaign_number, COUNT_DISTINCT(complaint_id, recall_row_id))",
            len(matched_pairs),
            len(grouped),
            output=grouped,
        )
        ordered = grouped.sort_values(
            ["matched_pair_count", "campaign_number"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(matched_pair_count DESC, campaign_number ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered[["campaign_number", "matched_pair_count"]])

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
