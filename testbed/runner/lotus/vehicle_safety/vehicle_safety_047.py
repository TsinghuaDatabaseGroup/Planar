#!/usr/bin/env python3
"""
vehicle_safety-047
Summarize Dodge electrical recall campaigns using semantically matching
complaints, complaint-to-recall timing, shared root cause, and promptness.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_FILTER -> SEM_EXTRACT ->
     GROUP_BY(campaign) -> SEM_AGGREGATE(promptness, dominant root cause) -> PROJECT
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
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-047"


def collect_values(values):
    return [value for value in values if pd.notna(value)]


def rounded_median(values):
    cleaned = values.dropna()
    if cleaned.empty:
        return None
    return int(round(float(cleaned.median())))


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
            "response_promptness_label": "",
            "dominant_electrical_root_cause": text,
        }
    if not isinstance(parsed, dict):
        return {
            "response_promptness_label": "",
            "dominant_electrical_root_cause": text,
        }
    return {
        "response_promptness_label": str(
            parsed.get("response_promptness_label", "")
        )
        .strip()
        .strip(".")
        .lower(),
        "dominant_electrical_root_cause": str(
            parsed.get("dominant_electrical_root_cause", "")
        ).strip(),
    }


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            [
                "complaint_id",
                "make",
                "component_id",
                "received_date",
                "summary_text",
            ]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        dodge_electrical_complaints = complaints[
            (complaints["make"] == "DODGE")
            & (complaints["component_id"] == "ELECTRICAL")
        ]
        tracker.record(
            "FILTER(make='DODGE' AND component_id='ELECTRICAL')",
            len(complaints),
            len(dodge_electrical_complaints),
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "campaign_report_received_date",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "component_component_id": "component_id",
                "risk_defect_summary": "recall_defect_summary",
            }
        )
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        dodge_electrical_recalls = recalls[
            (recalls["vehicle_make"] == "DODGE")
            & (recalls["component_id"] == "ELECTRICAL")
        ]
        tracker.record(
            "FILTER(vehicle.make='DODGE' AND component.component_id='ELECTRICAL')",
            len(recalls),
            len(dodge_electrical_recalls),
        )

        complaint_bindings = dodge_electrical_complaints.drop(
            columns=["make", "component_id"]
        ).copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=ELECTRICAL; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN complaint inputs)",
            len(dodge_electrical_complaints),
            len(complaint_bindings),
        )

        recall_bindings = dodge_electrical_recalls[
            [
                "campaign_number",
                "campaign_report_received_date",
                "recall_defect_summary",
            ]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=ELECTRICAL; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN recall inputs)",
            len(dodge_electrical_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Dodge ELECTRICAL complaint and recall share failure)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            if complaint_bindings.empty or recall_bindings.empty:
                matched_pairs = pd.DataFrame(
                    columns=[
                        "complaint_id",
                        "received_date",
                        "summary_text",
                        "campaign_number",
                        "campaign_report_received_date",
                        "recall_defect_summary",
                    ]
                )
            else:
                matched_pairs = complaint_bindings.sem_join(
                    recall_bindings,
                    "Within Dodge ELECTRICAL records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same electrical failure."
                )
            step.set_output(matched_pairs)

        with tracker.step(
            "SEM_FILTER(electrical failure with safety consequence)",
            input_rows=len(matched_pairs),
        ) as step:
            if matched_pairs.empty:
                safety_pairs = matched_pairs.copy()
            else:
                safety_pairs = matched_pairs.sem_filter(
                    "The matched complaint {summary_text} and recall defect "
                    "{recall_defect_summary} describe an electrical failure with "
                    "a safety consequence such as stalling, fire risk, loss of "
                    "lighting, loss of motive power, or warning-system failure."
                )
            step.set_output(safety_pairs)

        with tracker.step(
            "SEM_EXTRACT(electrical_root_cause<=8 words)",
            input_rows=len(safety_pairs),
        ) as step:
            if safety_pairs.empty:
                extracted = safety_pairs.copy()
                extracted["electrical_root_cause"] = pd.Series(dtype="object")
            else:
                extracted = safety_pairs.sem_extract(
                    input_cols=["summary_text", "recall_defect_summary"],
                    output_cols={
                        "electrical_root_cause": (
                            "The electrical root-cause phrase shared by the "
                            "complaint and recall in at most 8 words."
                        )
                    },
                )
                extracted["electrical_root_cause"] = (
                    extracted["electrical_root_cause"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(extracted)

        aggregation_input = extracted.copy()
        aggregation_input["_complaint_date"] = pd.to_datetime(
            aggregation_input["received_date"], errors="coerce"
        )
        aggregation_input["_recall_date"] = pd.to_datetime(
            aggregation_input["campaign_report_received_date"], errors="coerce"
        )
        aggregation_input["_days_to_recall"] = (
            aggregation_input["_recall_date"]
            - aggregation_input["_complaint_date"]
        ).dt.days
        aggregation_input["_before_recall"] = (
            aggregation_input["_complaint_date"]
            < aggregation_input["_recall_date"]
        )

        if aggregation_input.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "median_days_complaint_to_recall",
                    "share_complaints_before_recall",
                    "days_distribution",
                    "root_causes",
                    "complaint_summaries",
                ]
            )
        else:
            grouped = aggregation_input.groupby(
                "campaign_number", as_index=False
            ).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                median_days_complaint_to_recall=(
                    "_days_to_recall",
                    rounded_median,
                ),
                share_complaints_before_recall=("_before_recall", "mean"),
                days_distribution=("_days_to_recall", collect_values),
                root_causes=("electrical_root_cause", collect_values),
                complaint_summaries=("summary_text", collect_values),
            )
        tracker.record(
            "GROUP_BY([campaign.number], matched count, median days, before-recall share, distributions)",
            len(aggregation_input),
            len(grouped),
        )

        with tracker.step(
            "SEM_AGGREGATE(response promptness and dominant electrical root cause)",
            input_rows=len(grouped),
        ) as step:
            if grouped.empty:
                aggregate_output = pd.DataFrame(
                    columns=["campaign_number", "_aggregate_output"]
                )
            else:
                aggregate_output = grouped[
                    [
                        "campaign_number",
                        "days_distribution",
                        "root_causes",
                        "complaint_summaries",
                    ]
                ].sem_agg(
                    "Using days distribution {days_distribution}, extracted root "
                    "causes {root_causes}, and matched complaint descriptions "
                    "{complaint_summaries}, return only a JSON object with two "
                    "string fields: response_promptness_label and "
                    "dominant_electrical_root_cause. response_promptness_label "
                    "must be exactly one of fast, typical, or slow.",
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
                "response_promptness_label",
                "dominant_electrical_root_cause",
            ],
        )

        result = grouped.merge(parsed_aggregates, on="campaign_number", how="inner")[
            [
                "campaign_number",
                "matched_complaint_count",
                "median_days_complaint_to_recall",
                "share_complaints_before_recall",
                "dominant_electrical_root_cause",
                "response_promptness_label",
            ]
        ]
        tracker.record(
            "PROJECT([campaign_number, matched count, median days, before-recall share, dominant root cause, promptness])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
