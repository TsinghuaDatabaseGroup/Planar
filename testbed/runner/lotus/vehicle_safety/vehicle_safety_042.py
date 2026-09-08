#!/usr/bin/env python3
"""
vehicle_safety-042
Summarize Tesla ADAS recall campaigns supported by matching complaint narratives
and ODI investigation reports discussing the same driver-assistance issue.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_FILTER ->
     SEM_CLASSIFY(topic) -> SEM_JOIN(report FILTER) -> GROUP_BY(campaign) -> PROJECT
Output: table, metric: table_f1
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (
    StepTracker,
    Timer,
    df_records,
    load_docs,
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-042"


def alpha_ascending_mode(values):
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    max_count = counts.max()
    return sorted(str(value) for value in counts[counts == max_count].index)[0]


def distinct_sorted(values):
    return sorted({str(value) for value in values.dropna()})


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        tesla_adas_complaints = complaints[
            (complaints["make"] == "TESLA")
            & (complaints["component_id"] == "ADAS")
        ]
        tracker.record(
            "FILTER(make='TESLA' AND component_id='ADAS')",
            len(complaints),
            len(tesla_adas_complaints),
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
                "risk_consequence_summary",
            ]
        ].rename(
            columns={
                "component_component_id": "component_id",
                "risk_defect_summary": "recall_defect_summary",
                "risk_consequence_summary": "recall_consequence_summary",
            }
        )
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        tesla_adas_recalls = recalls[
            (recalls["vehicle_make"] == "TESLA")
            & (recalls["component_id"] == "ADAS")
        ]
        tracker.record(
            "FILTER(vehicle.make='TESLA' AND component.component_id='ADAS')",
            len(recalls),
            len(tesla_adas_recalls),
        )

        complaint_bindings = tesla_adas_complaints[
            ["complaint_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=ADAS; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN complaint inputs)",
            len(tesla_adas_complaints),
            len(complaint_bindings),
        )

        recall_bindings = tesla_adas_recalls[
            [
                "campaign_number",
                "recall_defect_summary",
                "recall_consequence_summary",
            ]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=ADAS; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN recall inputs)",
            len(tesla_adas_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Tesla ADAS complaint and recall share issue)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            if complaint_bindings.empty or recall_bindings.empty:
                complaint_recall_pairs = pd.DataFrame(
                    columns=[
                        "complaint_id",
                        "summary_text",
                        "campaign_number",
                        "recall_defect_summary",
                        "recall_consequence_summary",
                    ]
                )
            else:
                complaint_recall_pairs = complaint_bindings.sem_join(
                    recall_bindings,
                    "Within Tesla ADAS records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same driver-assistance or automation-control problem."
                )
            step.set_output(complaint_recall_pairs)

        with tracker.step(
            "SEM_FILTER(driving-context ADAS safety issue)",
            input_rows=len(complaint_recall_pairs),
        ) as step:
            if complaint_recall_pairs.empty:
                driving_issues = complaint_recall_pairs.copy()
            else:
                driving_issues = complaint_recall_pairs.sem_filter(
                    "The complaint {summary_text} describes a driving-context ADAS "
                    "safety issue rather than general infotainment or account "
                    "software behavior."
                )
            step.set_output(driving_issues)

        with tracker.step(
            "SEM_CLASSIFY(dominant Tesla ADAS defect topic)",
            input_rows=len(driving_issues),
        ) as step:
            if driving_issues.empty:
                topic_labeled = driving_issues.copy()
                topic_labeled["adas_topic"] = pd.Series(dtype="object")
            else:
                topic_labeled = driving_issues.sem_map(
                    "Assign the dominant Tesla ADAS defect topic for complaint "
                    "{summary_text}. Output exactly one label: autopilot_misuse, "
                    "unexpected_braking, steering_control, "
                    "warning_or_supervision_gap, or other.",
                    suffix="adas_topic",
                )
                topic_labeled["adas_topic"] = (
                    topic_labeled["adas_topic"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(topic_labeled)

        reports = load_docs(
            "nhtsa_vehicle_safety", "investigation_reports"
        ).rename(columns={"doc_id": "file_id", "contents": "body"})
        reports = reports[reports["file_id"].str.lower().str.endswith(".txt")]
        tracker.record(
            "SCAN_DOCS(investigation_reports, selector='*.txt')",
            None,
            len(reports),
        )

        candidate_reports = reports[
            reports["body"].str.contains(
                r"Tesla|Autopilot|driver assistance",
                case=False,
                na=False,
                regex=True,
            )
        ]
        tracker.record(
            "FILTER(body CONTAINS 'Tesla' OR body CONTAINS 'Autopilot' OR body CONTAINS 'driver assistance')",
            len(reports),
            len(candidate_reports),
        )

        issue_bindings = topic_labeled.copy()
        issue_bindings["issue_report_binding"] = (
            "topic="
            + issue_bindings["adas_topic"].fillna("").astype(str)
            + "; complaint_text="
            + issue_bindings["summary_text"].fillna("").astype(str)
            + "; recall_text="
            + issue_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind second SEM_JOIN issue inputs)",
            len(topic_labeled),
            len(issue_bindings),
        )

        with tracker.step(
            "SEM_JOIN(ADAS issue and report share safety or human-factors concern)",
            input_rows={
                "left": len(issue_bindings),
                "right": len(candidate_reports),
            },
        ) as step:
            if issue_bindings.empty or candidate_reports.empty:
                report_matches = pd.DataFrame(
                    columns=[
                        "complaint_id",
                        "campaign_number",
                        "adas_topic",
                        "file_id",
                    ]
                )
            else:
                report_matches = issue_bindings.sem_join(
                    candidate_reports[["file_id", "body"]],
                    "Match Tesla ADAS issue {issue_report_binding} to "
                    "investigation report {body} only when the report discusses "
                    "the same driver-assistance safety issue or ODI human-factors "
                    "concern."
                )
            step.set_output(report_matches)

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "dominant_adas_topic",
                    "related_report_ids",
                ]
            )
        else:
            grouped = report_matches.groupby("campaign_number", as_index=False).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                dominant_adas_topic=("adas_topic", alpha_ascending_mode),
                related_report_ids=("file_id", distinct_sorted),
            )
        tracker.record(
            "GROUP_BY([campaign.number], matched_complaint_count, dominant_adas_topic, related_report_ids)",
            len(report_matches),
            len(grouped),
        )

        result = grouped[
            [
                "campaign_number",
                "matched_complaint_count",
                "dominant_adas_topic",
                "related_report_ids",
            ]
        ]
        tracker.record(
            "PROJECT([campaign_number, matched_complaint_count, dominant_adas_topic, related_report_ids])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
