#!/usr/bin/env python3
"""
vehicle_safety-046
Summarize Chrysler airbag recall campaigns supported by matching complaints and
ODI investigation reports discussing the same airbag risk.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_FILTER ->
     SEM_CLASSIFY(failure mode) -> SEM_JOIN(report FILTER) ->
     GROUP_BY(campaign) -> SEM_AGGREGATE(remedy style, ODI evidence) -> PROJECT
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

TASK_ID = "vehicle_safety-046"


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
        return {"dominant_remedy_style": "", "odi_evidence_summary": text}
    if not isinstance(parsed, dict):
        return {"dominant_remedy_style": "", "odi_evidence_summary": text}
    return {
        "dominant_remedy_style": str(
            parsed.get("dominant_remedy_style", "")
        )
        .strip()
        .strip(".")
        .lower(),
        "odi_evidence_summary": str(
            parsed.get("odi_evidence_summary", "")
        ).strip(),
    }


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        chrysler_airbag_complaints = complaints[
            (complaints["make"] == "CHRYSLER")
            & (complaints["component_id"] == "AIRBAG")
        ]
        tracker.record(
            "FILTER(make='CHRYSLER' AND component_id='AIRBAG')",
            len(complaints),
            len(chrysler_airbag_complaints),
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
                "risk_consequence_summary",
                "remedy_corrective_action",
            ]
        ].rename(
            columns={
                "component_component_id": "component_id",
                "risk_defect_summary": "recall_defect_summary",
                "risk_consequence_summary": "recall_consequence_summary",
                "remedy_corrective_action": "recall_corrective_action",
            }
        )
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        chrysler_airbag_recalls = recalls[
            (recalls["vehicle_make"] == "CHRYSLER")
            & (recalls["component_id"] == "AIRBAG")
        ]
        tracker.record(
            "FILTER(vehicle.make='CHRYSLER' AND component.component_id='AIRBAG')",
            len(recalls),
            len(chrysler_airbag_recalls),
        )

        complaint_bindings = chrysler_airbag_complaints[
            ["complaint_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=AIRBAG; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN complaint inputs)",
            len(chrysler_airbag_complaints),
            len(complaint_bindings),
        )

        recall_bindings = chrysler_airbag_recalls[
            [
                "campaign_number",
                "recall_defect_summary",
                "recall_consequence_summary",
                "recall_corrective_action",
            ]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=AIRBAG; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN recall inputs)",
            len(chrysler_airbag_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Chrysler AIRBAG complaint and recall share defect)",
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
                        "recall_corrective_action",
                    ]
                )
            else:
                complaint_recall_pairs = complaint_bindings.sem_join(
                    recall_bindings,
                    "Within Chrysler AIRBAG records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same airbag defect."
                )
            step.set_output(complaint_recall_pairs)

        with tracker.step(
            "SEM_FILTER(airbag safety risk)",
            input_rows=len(complaint_recall_pairs),
        ) as step:
            if complaint_recall_pairs.empty:
                safety_pairs = complaint_recall_pairs.copy()
            else:
                safety_pairs = complaint_recall_pairs.sem_filter(
                    "The matched complaint {summary_text}, recall defect "
                    "{recall_defect_summary}, and consequence "
                    "{recall_consequence_summary} describe an airbag safety risk "
                    "such as non-deployment, unintended deployment, inflator "
                    "rupture, or occupant injury risk."
                )
            step.set_output(safety_pairs)

        with tracker.step(
            "SEM_CLASSIFY(dominant airbag failure mode)",
            input_rows=len(safety_pairs),
        ) as step:
            if safety_pairs.empty:
                failure_labeled = safety_pairs.copy()
                failure_labeled["airbag_failure_mode"] = pd.Series(dtype="object")
            else:
                failure_labeled = safety_pairs.sem_map(
                    "Assign the dominant airbag failure mode for complaint "
                    "{summary_text}. Output exactly one label: non_deployment, "
                    "unintended_deployment, inflator_rupture, "
                    "warning_or_sensor_fault, or other.",
                    suffix="airbag_failure_mode",
                )
                failure_labeled["airbag_failure_mode"] = (
                    failure_labeled["airbag_failure_mode"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(failure_labeled)

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
                r"Chrysler|airbag|inflator",
                case=False,
                na=False,
                regex=True,
            )
        ]
        tracker.record(
            "FILTER(body CONTAINS 'Chrysler' OR body CONTAINS 'airbag' OR body CONTAINS 'inflator')",
            len(reports),
            len(candidate_reports),
        )

        issue_bindings = failure_labeled.copy()
        issue_bindings["issue_report_binding"] = (
            "failure_mode="
            + issue_bindings["airbag_failure_mode"].fillna("").astype(str)
            + "; complaint_text="
            + issue_bindings["summary_text"].fillna("").astype(str)
            + "; recall_text="
            + issue_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind second SEM_JOIN issue inputs)",
            len(failure_labeled),
            len(issue_bindings),
        )

        with tracker.step(
            "SEM_JOIN(airbag issue and report share risk or ODI evidence)",
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
                        "airbag_failure_mode",
                        "recall_corrective_action",
                        "file_id",
                        "body",
                    ]
                )
            else:
                report_matches = issue_bindings.sem_join(
                    candidate_reports[["file_id", "body"]],
                    "Match Chrysler airbag issue {issue_report_binding} to "
                    "investigation report {body} only when the report discusses "
                    "the same airbag risk or ODI evidence."
                )
            step.set_output(report_matches)

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "dominant_airbag_failure_mode",
                    "corrective_actions",
                    "related_report_ids",
                    "report_bodies",
                ]
            )
        else:
            grouped = report_matches.groupby("campaign_number", as_index=False).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                dominant_airbag_failure_mode=(
                    "airbag_failure_mode",
                    alpha_ascending_mode,
                ),
                corrective_actions=("recall_corrective_action", distinct_in_order),
                related_report_ids=("file_id", distinct_sorted),
                report_bodies=("body", distinct_in_order),
            )
        tracker.record(
            "GROUP_BY([campaign.number], matched count, failure mode, remedies, report IDs and bodies)",
            len(report_matches),
            len(grouped),
        )

        with tracker.step(
            "SEM_AGGREGATE(dominant remedy style and ODI evidence by campaign)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                aggregate_output = pd.DataFrame(
                    columns=["campaign_number", "_aggregate_output"]
                )
            else:
                aggregate_output = grouped[
                    ["campaign_number", "corrective_actions", "report_bodies"]
                ].sem_agg(
                    "Using corrective actions {corrective_actions} and related ODI "
                    "report bodies {report_bodies}, return only a JSON object with "
                    "two string fields: dominant_remedy_style and "
                    "odi_evidence_summary. dominant_remedy_style must be exactly "
                    "one of software_update, part_replacement, or other; the ODI "
                    "evidence summary must be one short phrase.",
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
                "dominant_remedy_style",
                "odi_evidence_summary",
            ],
        )

        result = grouped.merge(parsed_aggregates, on="campaign_number", how="inner")[
            [
                "campaign_number",
                "matched_complaint_count",
                "dominant_airbag_failure_mode",
                "dominant_remedy_style",
                "related_report_ids",
                "odi_evidence_summary",
            ]
        ]
        tracker.record(
            "PROJECT([campaign_number, matched count, failure mode, remedy style, report IDs, ODI evidence])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
