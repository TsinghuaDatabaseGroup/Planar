#!/usr/bin/env python3
"""
vehicle_safety-036
Rank Ford airbag recall campaigns by semantically matching complaints and
summarize complaint severity, severe indicators, and remedy style.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_FILTER ->
     SEM_CLASSIFY(severity) -> PROJECT -> SEM_CLASSIFY(remedy) -> PROJECT ->
     GROUP_BY(campaign) -> ORDER_BY -> LIMIT -> PROJECT(rank)
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

TASK_ID = "vehicle_safety-036"


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

        ford_airbag_complaints = complaints[
            (complaints["make"] == "FORD")
            & (complaints["component_id"] == "AIRBAG")
        ]
        tracker.record(
            "FILTER(make='FORD' AND component_id='AIRBAG')",
            len(complaints),
            len(ford_airbag_complaints),
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
                "remedy_corrective_action",
            ]
        ].rename(
            columns={
                "component_component_id": "component_id",
                "risk_defect_summary": "recall_defect_summary",
                "remedy_corrective_action": "recall_corrective_action",
            }
        )
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        ford_airbag_recalls = recalls[
            (recalls["vehicle_make"] == "FORD")
            & (recalls["component_id"] == "AIRBAG")
        ]
        tracker.record(
            "FILTER(vehicle.make='FORD' AND component.component_id='AIRBAG')",
            len(recalls),
            len(ford_airbag_recalls),
        )

        complaint_bindings = ford_airbag_complaints.drop(
            columns=["make", "component_id"]
        ).copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=AIRBAG; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN complaint inputs)",
            len(ford_airbag_complaints),
            len(complaint_bindings),
        )

        recall_bindings = ford_airbag_recalls[
            ["campaign_number", "recall_defect_summary", "recall_corrective_action"]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=AIRBAG; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN recall inputs)",
            len(ford_airbag_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Ford AIRBAG complaint and recall share failure mechanism)",
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
                        "recall_corrective_action",
                    ]
                )
            else:
                matched_pairs = complaint_bindings.sem_join(
                    recall_bindings,
                    "Within Ford AIRBAG records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same airbag failure mechanism."
                )
            step.set_output(matched_pairs)

        with tracker.step(
            "SEM_FILTER(concrete airbag safety event)",
            input_rows=len(matched_pairs),
        ) as step:
            if matched_pairs.empty:
                concrete_events = matched_pairs.copy()
            else:
                concrete_events = matched_pairs.sem_filter(
                    "The complaint narrative {summary_text} describes a concrete "
                    "airbag safety event rather than a cosmetic, warning-light-only "
                    "concern."
                )
            step.set_output(concrete_events)

        with tracker.step(
            "SEM_CLASSIFY(complaint_severity)",
            input_rows=len(concrete_events),
        ) as step:
            if concrete_events.empty:
                severity_labeled = concrete_events.copy()
                severity_labeled["complaint_severity"] = pd.Series(dtype="object")
            else:
                severity_labeled = concrete_events.sem_map(
                    "Classify complaint {summary_text} severity. Output exactly one "
                    "label: minor_inconvenience, moderate_safety_event, or "
                    "severe_safety_event.",
                    suffix="complaint_severity",
                )
                severity_labeled["complaint_severity"] = (
                    severity_labeled["complaint_severity"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(severity_labeled)

        severity_projection = severity_labeled[
            [
                "complaint_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "summary_text",
                "campaign_number",
                "recall_corrective_action",
                "complaint_severity",
            ]
        ]
        tracker.record(
            "PROJECT([complaint fields, campaign.number, remedy.corrective_action, complaint_severity])",
            len(severity_labeled),
            len(severity_projection),
        )

        with tracker.step(
            "SEM_CLASSIFY(recall_remedy_style)",
            input_rows=len(severity_projection),
        ) as step:
            if severity_projection.empty:
                remedy_labeled = severity_projection.copy()
                remedy_labeled["recall_remedy_style"] = pd.Series(dtype="object")
            else:
                remedy_labeled = severity_projection.sem_map(
                    "Classify recall remedy {recall_corrective_action}. Output "
                    "exactly one label: software_update, part_replacement, or other.",
                    suffix="recall_remedy_style",
                )
                remedy_labeled["recall_remedy_style"] = (
                    remedy_labeled["recall_remedy_style"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(remedy_labeled)

        aggregation_input = remedy_labeled[
            [
                "complaint_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "campaign_number",
                "complaint_severity",
                "recall_remedy_style",
            ]
        ].copy()
        tracker.record(
            "PROJECT([complaint indicators, campaign.number, complaint_severity, recall_remedy_style])",
            len(remedy_labeled),
            len(aggregation_input),
        )

        aggregation_input["severe_indicator"] = (
            aggregation_input["crash_flag"]
            | (aggregation_input["injury_count"] > 0)
            | aggregation_input["vehicle_towed_flag"]
        )
        if aggregation_input.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "severe_indicator_rate",
                    "dominant_complaint_severity",
                    "dominant_recall_remedy_style",
                ]
            )
        else:
            grouped = aggregation_input.groupby(
                "campaign_number", as_index=False
            ).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                severe_indicator_rate=("severe_indicator", "mean"),
                dominant_complaint_severity=(
                    "complaint_severity",
                    alpha_ascending_mode,
                ),
                dominant_recall_remedy_style=(
                    "recall_remedy_style",
                    alpha_ascending_mode,
                ),
            )
        tracker.record(
            "GROUP_BY([campaign.number], matched_complaint_count, severe_indicator_rate, dominant severities)",
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

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("LIMIT(5)", len(ordered), len(top_five))

        result = top_five.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result = result[
            [
                "rank",
                "campaign_number",
                "matched_complaint_count",
                "severe_indicator_rate",
                "dominant_complaint_severity",
                "dominant_recall_remedy_style",
            ]
        ]
        tracker.record(
            "PROJECT([rank, campaign_number, matched_complaint_count, severe_indicator_rate, dominant labels])",
            len(top_five),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
