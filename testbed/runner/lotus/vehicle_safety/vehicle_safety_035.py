#!/usr/bin/env python3
"""
vehicle_safety-035
Characterize Dodge airbag recall campaigns that match both complaint narratives
and ODI investigation reports describing the same safety problem.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_FILTER -> SEM_EXTRACT ->
     SEM_JOIN(report FILTER) -> GROUP_BY(campaign) -> SEM_AGGREGATE -> PROJECT
Output: table, metric: f1
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

TASK_ID = "vehicle_safety-035"


def alpha_ascending_mode(values):
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    max_count = counts.max()
    return sorted(str(value) for value in counts[counts == max_count].index)[0]


def distinct_sorted(values):
    return sorted({str(value) for value in values.dropna()})


def distinct_in_order(values):
    return list(dict.fromkeys(str(value) for value in values.dropna()))


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        dodge_airbag_complaints = complaints[
            (complaints["make"] == "DODGE")
            & (complaints["component_id"] == "AIRBAG")
        ]
        tracker.record(
            "FILTER(make='DODGE' AND component_id='AIRBAG')",
            len(complaints),
            len(dodge_airbag_complaints),
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

        dodge_airbag_recalls = recalls[
            (recalls["vehicle_make"] == "DODGE")
            & (recalls["component_id"] == "AIRBAG")
        ]
        tracker.record(
            "FILTER(vehicle.make='DODGE' AND component.component_id='AIRBAG')",
            len(recalls),
            len(dodge_airbag_recalls),
        )

        complaint_bindings = dodge_airbag_complaints[
            ["complaint_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=AIRBAG; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN complaint inputs)",
            len(dodge_airbag_complaints),
            len(complaint_bindings),
        )

        recall_bindings = dodge_airbag_recalls[
            [
                "campaign_number",
                "recall_defect_summary",
                "recall_consequence_summary",
            ]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=AIRBAG; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN recall inputs)",
            len(dodge_airbag_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Dodge AIRBAG complaint and recall share failure mechanism)",
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
                    "Within Dodge AIRBAG records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same airbag failure mechanism or deployment hazard."
                )
            step.set_output(complaint_recall_pairs)

        with tracker.step(
            "SEM_FILTER(matched pair describes airbag safety consequence)",
            input_rows=len(complaint_recall_pairs),
        ) as step:
            if complaint_recall_pairs.empty:
                safety_pairs = complaint_recall_pairs.copy()
            else:
                safety_pairs = complaint_recall_pairs.sem_filter(
                    "The matched complaint {summary_text}, recall defect "
                    "{recall_defect_summary}, and recall consequence "
                    "{recall_consequence_summary} describe an airbag safety "
                    "consequence such as non-deployment, unintended deployment, "
                    "inflator rupture, injury, or crash exposure."
                )
            step.set_output(safety_pairs)

        with tracker.step(
            "SEM_EXTRACT(shared root_cause_phrase)",
            input_rows=len(safety_pairs),
        ) as step:
            if safety_pairs.empty:
                extracted_pairs = safety_pairs.copy()
                extracted_pairs["root_cause_phrase"] = pd.Series(dtype="object")
            else:
                extracted_pairs = safety_pairs.sem_extract(
                    input_cols=["summary_text", "recall_defect_summary"],
                    output_cols={
                        "root_cause_phrase": (
                            "A short root-cause phrase shared by the complaint and "
                            "recall defect descriptions."
                        )
                    },
                )
            step.set_output(extracted_pairs)

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
                r"Dodge|airbag|inflator",
                case=False,
                na=False,
                regex=True,
            )
        ]
        tracker.record(
            "FILTER(body CONTAINS 'Dodge' OR body CONTAINS 'airbag' OR body CONTAINS 'inflator')",
            len(reports),
            len(candidate_reports),
        )

        defect_bindings = extracted_pairs.copy()
        defect_bindings["defect_report_binding"] = (
            "root_cause="
            + defect_bindings["root_cause_phrase"].fillna("").astype(str)
            + "; complaint_text="
            + defect_bindings["summary_text"].fillna("").astype(str)
            + "; recall_text="
            + defect_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind second SEM_JOIN defect inputs)",
            len(extracted_pairs),
            len(defect_bindings),
        )

        with tracker.step(
            "SEM_JOIN(defect and report share airbag mechanism and ODI assessment)",
            input_rows={
                "left": len(defect_bindings),
                "right": len(candidate_reports),
            },
        ) as step:
            if defect_bindings.empty or candidate_reports.empty:
                report_matches = pd.DataFrame(
                    columns=[
                        "complaint_id",
                        "campaign_number",
                        "root_cause_phrase",
                        "file_id",
                        "body",
                    ]
                )
            else:
                report_matches = defect_bindings.sem_join(
                    candidate_reports[["file_id", "body"]],
                    "Match Dodge airbag defect {defect_report_binding} to "
                    "investigation report {body} only when the report discusses "
                    "the same airbag mechanism and ODI safety assessment."
                )
            step.set_output(report_matches)

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "related_report_ids",
                    "root_cause_phrase",
                    "report_bodies",
                ]
            )
        else:
            grouped = report_matches.groupby("campaign_number", as_index=False).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                related_report_ids=("file_id", distinct_sorted),
                root_cause_phrase=("root_cause_phrase", alpha_ascending_mode),
                report_bodies=("body", distinct_in_order),
            )
        tracker.record(
            "GROUP_BY([campaign.number], matched_complaint_count, related_report_ids, root_cause_phrase, report_bodies)",
            len(report_matches),
            len(grouped),
        )

        with tracker.step(
            "SEM_AGGREGATE(ODI safety conclusion by campaign)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                conclusions = pd.DataFrame(
                    columns=["campaign_number", "odi_safety_conclusion"]
                )
            else:
                conclusions = grouped[["campaign_number", "report_bodies"]].sem_agg(
                    "Summarize the ODI safety conclusion for this campaign from "
                    "the related report bodies {report_bodies} in one short phrase.",
                    suffix="odi_safety_conclusion",
                    group_by=["campaign_number"],
                )
            step.set_output(conclusions)

        result = grouped.merge(conclusions, on="campaign_number", how="inner")[
            [
                "campaign_number",
                "matched_complaint_count",
                "related_report_ids",
                "root_cause_phrase",
                "odi_safety_conclusion",
            ]
        ]
        tracker.record(
            "PROJECT([campaign_number, matched_complaint_count, related_report_ids, root_cause_phrase, odi_safety_conclusion])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
