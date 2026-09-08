#!/usr/bin/env python3
"""
vehicle_safety-037
Find the leading Acura engine root-cause theme supported by matching complaints,
recalls, and ODI investigation reports.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_EXTRACT -> SEM_FILTER ->
     SEM_JOIN(report FILTER) -> GROUP_BY(theme) -> SEM_AGGREGATE -> ORDER_BY ->
     LIMIT(1) -> PROJECT
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

TASK_ID = "vehicle_safety-037"


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

        acura_engine_complaints = complaints[
            (complaints["make"] == "ACURA")
            & (complaints["component_id"] == "ENGINE")
        ]
        tracker.record(
            "FILTER(make='ACURA' AND component_id='ENGINE')",
            len(complaints),
            len(acura_engine_complaints),
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

        acura_engine_recalls = recalls[
            (recalls["vehicle_make"] == "ACURA")
            & (recalls["component_id"] == "ENGINE")
        ]
        tracker.record(
            "FILTER(vehicle.make='ACURA' AND component.component_id='ENGINE')",
            len(recalls),
            len(acura_engine_recalls),
        )

        complaint_bindings = acura_engine_complaints[
            ["complaint_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=ENGINE; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN complaint inputs)",
            len(acura_engine_complaints),
            len(complaint_bindings),
        )

        recall_bindings = acura_engine_recalls[
            ["campaign_number", "recall_defect_summary", "recall_consequence_summary"]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=ENGINE; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind first SEM_JOIN recall inputs)",
            len(acura_engine_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Acura ENGINE complaint and recall share failure mechanism)",
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
                    "Within Acura ENGINE records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same engine failure mechanism."
                )
            step.set_output(complaint_recall_pairs)

        with tracker.step(
            "SEM_EXTRACT(root_cause_theme<=8 words)",
            input_rows=len(complaint_recall_pairs),
        ) as step:
            if complaint_recall_pairs.empty:
                extracted_pairs = complaint_recall_pairs.copy()
                extracted_pairs["root_cause_theme"] = pd.Series(dtype="object")
            else:
                extracted_pairs = complaint_recall_pairs.sem_extract(
                    input_cols=["summary_text", "recall_defect_summary"],
                    output_cols={
                        "root_cause_theme": (
                            "The shared engine root-cause theme in at most 8 words."
                        )
                    },
                )
            step.set_output(extracted_pairs)

        with tracker.step(
            "SEM_FILTER(driving-safety engine consequence)",
            input_rows=len(extracted_pairs),
        ) as step:
            if extracted_pairs.empty:
                safety_pairs = extracted_pairs.copy()
            else:
                safety_pairs = extracted_pairs.sem_filter(
                    "The matched complaint {summary_text} and recall consequence "
                    "{recall_consequence_summary} describe loss of motive power, "
                    "stalling, engine seizure, or another driving-safety consequence."
                )
            step.set_output(safety_pairs)

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
                r"Acura|Honda|engine failure",
                case=False,
                na=False,
                regex=True,
            )
        ]
        tracker.record(
            "FILTER(body CONTAINS 'Acura' OR body CONTAINS 'Honda' OR body CONTAINS 'engine failure')",
            len(reports),
            len(candidate_reports),
        )

        defect_bindings = safety_pairs.copy()
        defect_bindings["defect_report_binding"] = (
            "root_cause="
            + defect_bindings["root_cause_theme"].fillna("").astype(str)
            + "; complaint_text="
            + defect_bindings["summary_text"].fillna("").astype(str)
            + "; recall_text="
            + defect_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind second SEM_JOIN defect inputs)",
            len(safety_pairs),
            len(defect_bindings),
        )

        with tracker.step(
            "SEM_JOIN(root-cause theme and report share engine scope or ODI finding)",
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
                        "root_cause_theme",
                        "file_id",
                        "body",
                    ]
                )
            else:
                report_matches = defect_bindings.sem_join(
                    candidate_reports[["file_id", "body"]],
                    "Match Acura engine defect {defect_report_binding} to "
                    "investigation report {body} only when the report discusses "
                    "the same Acura or Honda engine-failure scope or ODI finding."
                )
            step.set_output(report_matches)

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "root_cause_theme",
                    "matched_complaint_count",
                    "distinct_campaign_count",
                    "related_report_ids",
                    "report_bodies",
                ]
            )
        else:
            grouped = report_matches.groupby("root_cause_theme", as_index=False).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                distinct_campaign_count=("campaign_number", "nunique"),
                related_report_ids=("file_id", distinct_sorted),
                report_bodies=("body", distinct_in_order),
            )
        tracker.record(
            "GROUP_BY([root_cause_theme], matched complaints, campaigns, report IDs, report bodies)",
            len(report_matches),
            len(grouped),
        )

        with tracker.step(
            "SEM_AGGREGATE(ODI evidence summary by root-cause theme)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                evidence_summaries = pd.DataFrame(
                    columns=["root_cause_theme", "odi_evidence_summary"]
                )
            else:
                evidence_summaries = grouped[
                    ["root_cause_theme", "report_bodies"]
                ].sem_agg(
                    "Summarize the ODI evidence across the related report bodies "
                    "{report_bodies} in one short phrase.",
                    suffix="odi_evidence_summary",
                    group_by=["root_cause_theme"],
                )
            step.set_output(evidence_summaries)

        with_evidence = grouped.merge(
            evidence_summaries,
            on="root_cause_theme",
            how="inner",
        )
        ordered = with_evidence.sort_values(
            ["matched_complaint_count", "root_cause_theme"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([matched_complaint_count DESC, root_cause_theme ASC])",
            len(with_evidence),
            len(ordered),
        )

        leading = ordered.head(1).reset_index(drop=True)
        tracker.record("LIMIT(1)", len(ordered), len(leading))

        result = leading[
            [
                "root_cause_theme",
                "matched_complaint_count",
                "distinct_campaign_count",
                "related_report_ids",
                "odi_evidence_summary",
            ]
        ]
        tracker.record(
            "PROJECT([root_cause_theme, matched_complaint_count, distinct_campaign_count, related_report_ids, odi_evidence_summary])",
            len(leading),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
