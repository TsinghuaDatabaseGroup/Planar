#!/usr/bin/env python3
"""LOTUS pipeline for vehicle_safety-014."""

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


TASK_ID = "vehicle_safety-014"


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
        dodge_airbag_complaints = complaints[
            (complaints["make"] == "DODGE")
            & (complaints["component_id"] == "AIRBAG")
        ].copy()
        tracker.record(
            "FILTER(make='DODGE' AND component_id='AIRBAG')",
            len(complaints),
            len(dodge_airbag_complaints),
            output=dodge_airbag_complaints,
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
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
        dodge_airbag_recalls = recalls[
            (recalls["make"] == "DODGE")
            & (recalls["component_id"] == "AIRBAG")
        ].copy()
        tracker.record(
            "FILTER(make='DODGE' AND component_id='AIRBAG')",
            len(recalls),
            len(dodge_airbag_recalls),
            output=dodge_airbag_recalls,
        )
        dedup_input_rows = len(dodge_airbag_recalls)
        dodge_airbag_recalls = dodge_airbag_recalls.drop_duplicates(
            subset=["campaign_number"],
            keep="first",
        )
        tracker.record(
            "DEDUP(campaign_number)",
            dedup_input_rows,
            len(dodge_airbag_recalls),
            output=dodge_airbag_recalls,
        )

        complaint_bindings = dodge_airbag_complaints[
            ["complaint_id", "summary_text"]
        ].rename(columns={"summary_text": "complaint_summary"})
        tracker.record(
            "PROJECT(complaint_id, complaint_summary)",
            len(dodge_airbag_complaints),
            len(complaint_bindings),
            output=complaint_bindings,
        )
        recall_bindings = dodge_airbag_recalls[
            ["campaign_number", "defect_summary"]
        ].copy()
        tracker.record(
            "PROJECT(campaign_number, defect_summary)",
            len(dodge_airbag_recalls),
            len(recall_bindings),
            output=recall_bindings,
        )

        with tracker.step(
            "SEM_JOIN(same concrete airbag failure or deployment hazard)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            matched_pairs = complaint_bindings.sem_join(
                recall_bindings,
                "Keep complaint {complaint_summary:left} and recall defect "
                "{defect_summary:right} together only when both describe the "
                "same concrete airbag failure mechanism or deployment hazard, "
                "such as the same non-deployment, unintended deployment, or "
                "inflator-rupture problem. A generic shared mention of airbags "
                "is insufficient."
            )
            step.set_output(matched_pairs)

        distinct_pairs = matched_pairs[
            ["complaint_id", "campaign_number"]
        ].drop_duplicates(ignore_index=True)
        grouped = (
            distinct_pairs.groupby("campaign_number", sort=False)
            .size()
            .rename("matched_pair_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(campaign_number, COUNT_DISTINCT(pair))",
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
