#!/usr/bin/env python3
"""
vehicle_safety-041
Summarize Tesla electrical recall campaigns supported by matching complaints
and ODI investigation reports discussing the same risk.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_EXTRACT -> SEM_FILTER ->
     SEM_JOIN(report FILTER) -> GROUP_BY(campaign) -> SEM_AGGREGATE -> PROJECT
Output: table, metric: f1
"""
import json
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

TASK_ID = "vehicle_safety-041"


def distinct_sorted(values):
    return sorted({str(value) for value in values.dropna()})


def collect_values(values):
    return [str(value) for value in values.dropna()]


def distinct_in_order(values):
    return list(dict.fromkeys(str(value) for value in values.dropna()))


def parse_aggregate_json(value) -> dict[str, str]:
    text = str(value).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {
            "dominant_defect_mechanism": "",
            "odi_risk_summary": text,
        }
    if not isinstance(parsed, dict):
        return {
            "dominant_defect_mechanism": "",
            "odi_risk_summary": text,
        }
    return {
        "dominant_defect_mechanism": str(
            parsed.get("dominant_defect_mechanism", "")
        ).strip(),
        "odi_risk_summary": str(parsed.get("odi_risk_summary", "")).strip(),
    }


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        tesla_electrical_complaints = complaints[
            (complaints["make"] == "TESLA")
            & (complaints["component_id"] == "ELECTRICAL")
        ]
        tracker.record(
            "FILTER(make='TESLA' AND component_id='ELECTRICAL')",
            len(complaints),
            len(tesla_electrical_complaints),
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

        tesla_electrical_recalls = recalls[
            (recalls["vehicle_make"] == "TESLA")
            & (recalls["component_id"] == "ELECTRICAL")
        ]
        tracker.record(
            "FILTER(vehicle.make='TESLA' AND component.component_id='ELECTRICAL')",
            len(recalls),
            len(tesla_electrical_recalls),
        )

        complaint_bindings = tesla_electrical_complaints[
            ["complaint_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=ELECTRICAL; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN complaint inputs)",
            len(tesla_electrical_complaints),
            len(complaint_bindings),
        )

        recall_bindings = tesla_electrical_recalls[
            ["campaign_number", "recall_defect_summary", "recall_consequence_summary"]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=ELECTRICAL; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN recall inputs)",
            len(tesla_electrical_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Tesla ELECTRICAL complaint and recall share defect)",
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
                    "Within Tesla ELECTRICAL records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same electrical, warning, software-control, or "
                    "power-loss defect."
                )
            step.set_output(complaint_recall_pairs)

        with tracker.step(
            "SEM_EXTRACT(defect_mechanism<=8 words)",
            input_rows=len(complaint_recall_pairs),
        ) as step:
            if complaint_recall_pairs.empty:
                extracted = complaint_recall_pairs.copy()
                extracted["defect_mechanism"] = pd.Series(dtype="object")
            else:
                extracted = complaint_recall_pairs.sem_extract(
                    input_cols=["summary_text", "recall_defect_summary"],
                    output_cols={
                        "defect_mechanism": (
                            "The dominant Tesla electrical defect mechanism in at "
                            "most 8 words, shared by the complaint and recall."
                        )
                    },
                )
                extracted["defect_mechanism"] = (
                    extracted["defect_mechanism"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(extracted)

        with tracker.step(
            "SEM_FILTER(driving-safety or warning-control risk)",
            input_rows=len(extracted),
        ) as step:
            if extracted.empty:
                safety_matches = extracted.copy()
            else:
                safety_matches = extracted.sem_filter(
                    "The matched complaint {summary_text} and recall consequence "
                    "{recall_consequence_summary} describe a driving-safety or "
                    "warning-control risk, not only an infotainment or convenience "
                    "issue."
                )
            step.set_output(safety_matches)

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
                r"Tesla|Autopilot|electrical",
                case=False,
                na=False,
                regex=True,
            )
        ]
        tracker.record(
            "FILTER(body CONTAINS 'Tesla' OR body CONTAINS 'Autopilot' OR body CONTAINS 'electrical')",
            len(reports),
            len(candidate_reports),
        )

        defect_bindings = safety_matches.copy()
        defect_bindings["defect_report_binding"] = (
            "mechanism="
            + defect_bindings["defect_mechanism"].fillna("").astype(str)
            + "; complaint_text="
            + defect_bindings["summary_text"].fillna("").astype(str)
            + "; recall_text="
            + defect_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind second SEM_JOIN defect inputs)",
            len(safety_matches),
            len(defect_bindings),
        )

        with tracker.step(
            "SEM_JOIN(defect and report share Tesla electrical risk and ODI assessment)",
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
                        "defect_mechanism",
                        "file_id",
                        "body",
                    ]
                )
            else:
                report_matches = defect_bindings.sem_join(
                    candidate_reports[["file_id", "body"]],
                    "Match Tesla electrical defect {defect_report_binding} to "
                    "investigation report {body} only when the report discusses "
                    "the same Tesla control, warning, or electrical-failure risk "
                    "and ODI assessment."
                )
            step.set_output(report_matches)

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "related_report_ids",
                    "defect_mechanisms",
                    "report_bodies",
                ]
            )
        else:
            grouped = report_matches.groupby("campaign_number", as_index=False).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                related_report_ids=("file_id", distinct_sorted),
                defect_mechanisms=("defect_mechanism", collect_values),
                report_bodies=("body", distinct_in_order),
            )
        tracker.record(
            "GROUP_BY([campaign.number], matched count, report IDs, mechanisms, report bodies)",
            len(report_matches),
            len(grouped),
        )

        with tracker.step(
            "SEM_AGGREGATE(ODI risk and dominant mechanism by campaign)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                aggregate_output = pd.DataFrame(
                    columns=["campaign_number", "_aggregate_output"]
                )
            else:
                aggregate_output = grouped[
                    ["campaign_number", "defect_mechanisms", "report_bodies"]
                ].sem_agg(
                    "Using defect mechanisms {defect_mechanisms} and related ODI "
                    "report bodies {report_bodies}, return only a JSON object with "
                    "two string fields: dominant_defect_mechanism and "
                    "odi_risk_summary. The ODI risk summary must be one short phrase.",
                    suffix="_aggregate_output",
                    group_by=["campaign_number"],
                )
            step.set_output(aggregate_output)

        parsed_rows = []
        for _, row in aggregate_output.iterrows():
            parsed = parse_aggregate_json(row["_aggregate_output"])
            parsed_rows.append(
                {
                    "campaign_number": row["campaign_number"],
                    **parsed,
                }
            )
        parsed_aggregates = pd.DataFrame(
            parsed_rows,
            columns=[
                "campaign_number",
                "dominant_defect_mechanism",
                "odi_risk_summary",
            ],
        )

        result = grouped.merge(parsed_aggregates, on="campaign_number", how="inner")[
            [
                "campaign_number",
                "matched_complaint_count",
                "related_report_ids",
                "dominant_defect_mechanism",
                "odi_risk_summary",
            ]
        ]
        tracker.record(
            "PROJECT([campaign_number, matched_complaint_count, related_report_ids, dominant_defect_mechanism, odi_risk_summary])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
