#!/usr/bin/env python3
"""LOTUS pipeline for vehicle_safety-013."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    load_jsonl,
    load_table,
    save_output,
    setup,
)


TASK_ID = "vehicle_safety-013"


def main():
    setup(max_tokens=512)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            [
                "complaint_id",
                "vehicle_id",
                "component_id",
                "received_date",
                "summary_text",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(complaints.csv)",
            None,
            len(complaints),
            output=complaints,
        )
        complaint_input_rows = len(complaints)
        complaints = complaints[
            complaints["received_date"].notna()
            & complaints["received_date"].astype(str).str.strip().ne("")
        ].copy()
        tracker.record(
            "FILTER(received_date IS NOT NULL AND received_date<>'')",
            complaint_input_rows,
            len(complaints),
            output=complaints,
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "recall_id",
                "campaign_number",
                "vehicle_vehicle_id",
                "component_component_id",
                "campaign_report_received_date",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "recall_vehicle_id",
                "component_component_id": "recall_component_id",
                "campaign_report_received_date": "recall_report_received_date",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record(
            "SCAN_DOCS(recalls.jsonl)",
            None,
            len(recalls),
            output=recalls,
        )
        recall_input_rows = len(recalls)
        recalls = recalls[
            recalls["recall_report_received_date"].notna()
            & recalls["recall_report_received_date"].astype(str).str.strip().ne("")
        ].copy()
        tracker.record(
            "FILTER(recall_report_received_date IS NOT NULL AND <> '')",
            recall_input_rows,
            len(recalls),
            output=recalls,
        )

        complaints["complaint_record"] = (
            "vehicle_id: "
            + complaints["vehicle_id"].astype(str)
            + "\ncomponent_id: "
            + complaints["component_id"].astype(str)
            + "\ncomplaint_received_date: "
            + complaints["received_date"].astype(str)
            + "\ncomplaint_summary: "
            + complaints["summary_text"].fillna("").astype(str)
        )
        complaint_bindings = complaints[
            ["complaint_id", "complaint_record"]
        ].copy()
        tracker.record(
            "PROJECT(complaint semantic binding)",
            len(complaints),
            len(complaint_bindings),
            output=complaint_bindings,
        )

        recalls["recall_record"] = (
            "vehicle_id: "
            + recalls["recall_vehicle_id"].astype(str)
            + "\ncomponent_id: "
            + recalls["recall_component_id"].astype(str)
            + "\nrecall_report_received_date: "
            + recalls["recall_report_received_date"].astype(str)
            + "\ncampaign_number: "
            + recalls["campaign_number"].astype(str)
            + "\nrecall_defect_summary: "
            + recalls["defect_summary"].fillna("").astype(str)
        )
        recall_bindings = recalls[["recall_id", "recall_record"]].copy()
        tracker.record(
            "PROJECT(recall semantic binding)",
            len(recalls),
            len(recall_bindings),
            output=recall_bindings,
        )

        with tracker.step(
            "SEM_JOIN(same eligible vehicle/component defect mechanism)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            matched_pairs = complaint_bindings.sem_join(
                recall_bindings,
                "For complaint {complaint_record:left} and recall "
                "{recall_record:right}, keep the pair only when they have the "
                "same vehicle_id and component_id, the recall report date is at "
                "least 90 days after the complaint received date, and the "
                "complaint summary and recall defect summary describe the same "
                "concrete underlying defect mechanism rather than merely the "
                "same broad component or symptom."
            )
            step.set_output(matched_pairs)

        distinct_pairs = matched_pairs[
            ["complaint_id", "recall_id"]
        ].drop_duplicates(ignore_index=True)
        answer = int(len(distinct_pairs))
        tracker.record(
            "GROUP_BY([], COUNT_DISTINCT(complaint_id, recall_id))",
            len(matched_pairs),
            1,
            output=answer,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
