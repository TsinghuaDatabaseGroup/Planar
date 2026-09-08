#!/usr/bin/env python3
"""
vehicle_safety-045
Rank Chevrolet braking-defect themes supported by matching complaints, recalls,
and ODI investigation reports.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_EXTRACT -> SEM_FILTER ->
     SEM_JOIN(report FILTER) -> GROUP_BY(theme) -> SEM_AGGREGATE -> ORDER_BY ->
     LIMIT(2) -> PROJECT(rank)
Output: ordered list, metric: ordered_f1
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

TASK_ID = "vehicle_safety-045"


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

        chevrolet_brake_complaints = complaints[
            (complaints["make"] == "CHEVROLET")
            & (complaints["component_id"] == "BRAKES")
        ]
        tracker.record(
            "FILTER(make='CHEVROLET' AND component_id='BRAKES')",
            len(complaints),
            len(chevrolet_brake_complaints),
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

        chevrolet_brake_recalls = recalls[
            (recalls["vehicle_make"] == "CHEVROLET")
            & (recalls["component_id"] == "BRAKES")
        ]
        tracker.record(
            "FILTER(vehicle.make='CHEVROLET' AND component.component_id='BRAKES')",
            len(recalls),
            len(chevrolet_brake_recalls),
        )

        complaint_bindings = chevrolet_brake_complaints[
            ["complaint_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=BRAKES; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN complaint inputs)",
            len(chevrolet_brake_complaints),
            len(complaint_bindings),
        )

        recall_bindings = chevrolet_brake_recalls[
            ["campaign_number", "recall_defect_summary", "recall_consequence_summary"]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=BRAKES; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN recall inputs)",
            len(chevrolet_brake_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Chevrolet BRAKES complaint and recall share defect or risk)",
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
                    "Within Chevrolet BRAKES records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same braking defect or risk."
                )
            step.set_output(complaint_recall_pairs)

        with tracker.step(
            "SEM_EXTRACT(shared defect_theme<=8 words)",
            input_rows=len(complaint_recall_pairs),
        ) as step:
            if complaint_recall_pairs.empty:
                extracted = complaint_recall_pairs.copy()
                extracted["defect_theme"] = pd.Series(dtype="object")
            else:
                extracted = complaint_recall_pairs.sem_extract(
                    input_cols=["summary_text", "recall_defect_summary"],
                    output_cols={
                        "defect_theme": (
                            "The braking defect theme shared by the complaint and "
                            "recall in at most 8 words."
                        )
                    },
                )
                extracted["defect_theme"] = (
                    extracted["defect_theme"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(extracted)

        with tracker.step(
            "SEM_FILTER(braking safety risk with serious consequence)",
            input_rows=len(extracted),
        ) as step:
            if extracted.empty:
                safety_matches = extracted.copy()
            else:
                safety_matches = extracted.sem_filter(
                    "The matched complaint {summary_text} and recall consequence "
                    "{recall_consequence_summary} describe a braking safety risk "
                    "with a potential crash, loss-of-control, or extended-stopping "
                    "consequence."
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
                r"Chevrolet|GM|brake",
                case=False,
                na=False,
                regex=True,
            )
        ]
        tracker.record(
            "FILTER(body CONTAINS 'Chevrolet' OR body CONTAINS 'GM' OR body CONTAINS 'brake')",
            len(reports),
            len(candidate_reports),
        )

        defect_bindings = safety_matches.copy()
        defect_bindings["defect_report_binding"] = (
            "theme="
            + defect_bindings["defect_theme"].fillna("").astype(str)
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
            "SEM_JOIN(defect theme and report share Chevrolet or GM braking risk)",
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
                        "defect_theme",
                        "file_id",
                        "body",
                    ]
                )
            else:
                report_matches = defect_bindings.sem_join(
                    candidate_reports[["file_id", "body"]],
                    "Match braking defect {defect_report_binding} to investigation "
                    "report {body} only when the report discusses the same "
                    "Chevrolet or GM braking issue or ODI risk finding."
                )
            step.set_output(report_matches)

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "defect_theme",
                    "matched_complaint_count",
                    "distinct_campaign_count",
                    "related_report_ids",
                    "report_bodies",
                ]
            )
        else:
            grouped = report_matches.groupby("defect_theme", as_index=False).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                distinct_campaign_count=("campaign_number", "nunique"),
                related_report_ids=("file_id", distinct_sorted),
                report_bodies=("body", distinct_in_order),
            )
        tracker.record(
            "GROUP_BY([defect_theme], matched complaints, campaigns, report IDs, report bodies)",
            len(report_matches),
            len(grouped),
        )

        with tracker.step(
            "SEM_AGGREGATE(ODI braking risk summary by defect theme)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                risk_summaries = pd.DataFrame(
                    columns=["defect_theme", "odi_risk_summary"]
                )
            else:
                risk_summaries = grouped[
                    ["defect_theme", "report_bodies"]
                ].sem_agg(
                    "Summarize the ODI braking risk finding from related report "
                    "bodies {report_bodies} for this defect theme in one short phrase.",
                    suffix="odi_risk_summary",
                    group_by=["defect_theme"],
                )
            step.set_output(risk_summaries)

        with_summaries = grouped.merge(
            risk_summaries,
            on="defect_theme",
            how="inner",
        )
        ordered = with_summaries.sort_values(
            ["matched_complaint_count", "defect_theme"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([matched_complaint_count DESC, defect_theme ASC])",
            len(with_summaries),
            len(ordered),
        )

        top_two = ordered.head(2).reset_index(drop=True)
        tracker.record("LIMIT(2)", len(ordered), len(top_two))

        result = top_two.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result = result[
            [
                "rank",
                "defect_theme",
                "matched_complaint_count",
                "distinct_campaign_count",
                "related_report_ids",
                "odi_risk_summary",
            ]
        ]
        tracker.record(
            "PROJECT([rank, defect_theme, matched_complaint_count, distinct_campaign_count, related_report_ids, odi_risk_summary])",
            len(top_two),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
