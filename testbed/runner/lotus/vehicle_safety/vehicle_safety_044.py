#!/usr/bin/env python3
"""
vehicle_safety-044
Rank the top three Acura brake recall campaigns by semantically matching
braking-performance complaints and summarize severity and failure mode.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_FILTER -> SEM_EXTRACT ->
     SEM_CLASSIFY(recall severity) -> GROUP_BY(campaign) -> ORDER_BY -> LIMIT ->
     PROJECT(rank)
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
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-044"


def alpha_ascending_mode(values):
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    max_count = counts.max()
    return sorted(str(value) for value in counts[counts == max_count].index)[0]


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            [
                "complaint_id",
                "make",
                "component_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "summary_text",
            ]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        acura_brake_complaints = complaints[
            (complaints["make"] == "ACURA")
            & (complaints["component_id"] == "BRAKES")
        ]
        tracker.record(
            "FILTER(make='ACURA' AND component_id='BRAKES')",
            len(complaints),
            len(acura_brake_complaints),
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

        acura_brake_recalls = recalls[
            (recalls["vehicle_make"] == "ACURA")
            & (recalls["component_id"] == "BRAKES")
        ]
        tracker.record(
            "FILTER(vehicle.make='ACURA' AND component.component_id='BRAKES')",
            len(recalls),
            len(acura_brake_recalls),
        )

        complaint_bindings = acura_brake_complaints.drop(
            columns=["make", "component_id"]
        ).copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=BRAKES; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN complaint inputs)",
            len(acura_brake_complaints),
            len(complaint_bindings),
        )

        recall_bindings = acura_brake_recalls[
            ["campaign_number", "recall_defect_summary", "recall_consequence_summary"]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=BRAKES; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN recall inputs)",
            len(acura_brake_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Acura BRAKES complaint and recall share defect)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            if complaint_bindings.empty or recall_bindings.empty:
                matched_pairs = pd.DataFrame(
                    columns=[
                        "complaint_id",
                        "crash_flag",
                        "injury_count",
                        "vehicle_towed_flag",
                        "summary_text",
                        "campaign_number",
                        "recall_defect_summary",
                        "recall_consequence_summary",
                    ]
                )
            else:
                matched_pairs = complaint_bindings.sem_join(
                    recall_bindings,
                    "Within Acura BRAKES records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same braking defect."
                )
            step.set_output(matched_pairs)

        with tracker.step(
            "SEM_FILTER(real braking-performance safety issue)",
            input_rows=len(matched_pairs),
        ) as step:
            if matched_pairs.empty:
                braking_issues = matched_pairs.copy()
            else:
                braking_issues = matched_pairs.sem_filter(
                    "The complaint {summary_text} describes a real "
                    "braking-performance safety issue such as loss of braking, "
                    "extended stopping distance, unintended braking, or brake failure."
                )
            step.set_output(braking_issues)

        with tracker.step(
            "SEM_EXTRACT(braking_failure_mode<=6 words)",
            input_rows=len(braking_issues),
        ) as step:
            if braking_issues.empty:
                extracted = braking_issues.copy()
                extracted["braking_failure_mode"] = pd.Series(dtype="object")
            else:
                extracted = braking_issues.sem_extract(
                    input_cols=["summary_text", "recall_defect_summary"],
                    output_cols={
                        "braking_failure_mode": (
                            "The dominant braking failure mode in at most 6 words, "
                            "shared by the complaint and recall."
                        )
                    },
                )
                extracted["braking_failure_mode"] = (
                    extracted["braking_failure_mode"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(extracted)

        with tracker.step(
            "SEM_CLASSIFY(recall_severity)",
            input_rows=len(extracted),
        ) as step:
            if extracted.empty:
                severity_labeled = extracted.copy()
                severity_labeled["recall_severity"] = pd.Series(dtype="object")
            else:
                severity_labeled = extracted.sem_map(
                    "Assign recall consequence {recall_consequence_summary} "
                    "severity. Output exactly one label: critical, severe, "
                    "moderate, or minor.",
                    suffix="recall_severity",
                )
                severity_labeled["recall_severity"] = (
                    severity_labeled["recall_severity"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(severity_labeled)

        aggregation_input = severity_labeled.copy()
        aggregation_input["severe_complaint"] = (
            aggregation_input["crash_flag"]
            | (aggregation_input["injury_count"] > 0)
            | aggregation_input["vehicle_towed_flag"]
        )
        if aggregation_input.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "severe_complaint_rate",
                    "dominant_braking_failure_mode",
                    "dominant_recall_severity",
                ]
            )
        else:
            grouped = aggregation_input.groupby(
                "campaign_number", as_index=False
            ).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                severe_complaint_rate=("severe_complaint", "mean"),
                dominant_braking_failure_mode=(
                    "braking_failure_mode",
                    alpha_ascending_mode,
                ),
                dominant_recall_severity=(
                    "recall_severity",
                    alpha_ascending_mode,
                ),
            )
        tracker.record(
            "GROUP_BY([campaign.number], matched count, severe rate, failure mode, recall severity)",
            len(aggregation_input),
            len(grouped),
        )

        ordered = grouped.sort_values(
            ["matched_complaint_count", "campaign_number"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([matched_complaint_count DESC, campaign.number ASC])",
            len(grouped),
            len(ordered),
        )

        top_three = ordered.head(3).reset_index(drop=True)
        tracker.record("LIMIT(3)", len(ordered), len(top_three))

        result = top_three.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result = result[
            [
                "rank",
                "campaign_number",
                "matched_complaint_count",
                "severe_complaint_rate",
                "dominant_braking_failure_mode",
                "dominant_recall_severity",
            ]
        ]
        tracker.record(
            "PROJECT([rank, campaign_number, matched_complaint_count, severe_complaint_rate, dominant modes])",
            len(top_three),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
