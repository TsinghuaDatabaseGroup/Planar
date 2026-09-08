#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-047."""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_jsonl,
    load_table,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-047"
DATASET = "nhtsa_vehicle_safety"
PROMPTNESS_LABELS = ("fast", "typical", "slow")
MATCH_COLUMNS = [
    "complaint_id",
    "component_id",
    "received_date",
    "summary_text",
    "campaign_number",
    "campaign_report_received_date",
    "recall_component_id",
    "recall_defect_summary",
]


def normalize_root_cause(value) -> str:
    return " ".join(str(value).strip().removesuffix(".").lower().split())


def collect_values(values: pd.Series) -> list:
    return values.dropna().tolist()


def rounded_median(values: pd.Series) -> int | None:
    cleaned = values.dropna()
    if cleaned.empty:
        return None
    return int(round(float(cleaned.median())))


def parse_campaign_summary(value) -> dict[str, str]:
    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:] if lines else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    promptness = normalize_enum(
        parsed.get("response_promptness_label", ""),
        PROMPTNESS_LABELS,
    )
    if promptness is None:
        raise ValueError(f"{TASK_ID}: invalid response promptness label")
    return {
        "response_promptness_label": promptness,
        "dominant_electrical_root_cause": str(
            parsed.get("dominant_electrical_root_cause", "")
        ).strip(),
    }


def build_campaign_summary_plan(index: int, row: dict):
    aggregate_input = pd.DataFrame(
        [
            {
                "days_distribution": row["days_distribution"],
                "root_causes": row["root_causes"],
                "complaint_summaries": row["complaint_summaries"],
            }
        ]
    )
    aggregate_plan = memory_dataset(
        f"{TASK_ID}-aggregate-{index}", aggregate_input
    ).sem_agg(
        col={
            "name": "campaign_summary_json",
            "type": str,
            "desc": (
                "A JSON-encoded string whose decoded object has exactly "
                "the string fields response_promptness_label and "
                "dominant_electrical_root_cause."
            ),
        },
        agg=(
            "Assign response_promptness_label as exactly one of fast, "
            "typical, or slow from the days distribution and matched "
            "complaint descriptions, and choose the dominant electrical "
            "root-cause phrase. Set the output field "
            "campaign_summary_json to a JSON-encoded string whose "
            "decoded object contains exactly the string fields "
            "response_promptness_label and "
            "dominant_electrical_root_cause. Do not place those two "
            "fields at the outer response level."
        ),
        depends_on=[
            "days_distribution",
            "root_causes",
            "complaint_summaries",
        ],
    )
    return aggregate_input, aggregate_plan


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            [
                "complaint_id",
                "make",
                "component_id",
                "received_date",
                "summary_text",
            ],
        )
        tracker.record("scan", None, complaints)

        dodge_complaints = complaints.loc[
            (complaints["make"] == "DODGE")
            & (complaints["component_id"] == "ELECTRICAL"),
            [
                "complaint_id",
                "component_id",
                "received_date",
                "summary_text",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), dodge_complaints)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "campaign_report_received_date",
            "vehicle_make",
            "component_component_id",
            "risk_defect_summary",
        ]].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
            }
        )
        tracker.record("scan", None, recalls)

        dodge_recalls = recalls.loc[
            (recalls["vehicle_make"] == "DODGE")
            & (recalls["recall_component_id"] == "ELECTRICAL"),
            [
                "campaign_number",
                "campaign_report_received_date",
                "recall_component_id",
                "recall_defect_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), dodge_recalls)

        match_plan = memory_dataset(
            f"{TASK_ID}-complaints", dodge_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", dodge_recalls),
            condition=(
                "Within Dodge ELECTRICAL records, match a complaint to a recall "
                "only when the complaint narrative and recall defect summary "
                "describe the same electrical failure."
            ),
            depends_on=MATCH_COLUMNS,
        )
        started = time.time()
        match_result = match_plan.run(config)
        matched_pairs = result_frame(match_result)
        if matched_pairs.empty:
            matched_pairs = pd.DataFrame(columns=MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(dodge_complaints), "right": len(dodge_recalls)},
            matched_pairs,
            match_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety", matched_pairs
        ).sem_filter(
            filter=(
                "The matched evidence describes an electrical failure with a "
                "safety consequence such as stalling, fire risk, loss of "
                "lighting, loss of motive power, or warning-system failure."
            ),
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        safety_pairs = result_frame(safety_result, matched_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(matched_pairs),
            safety_pairs,
            safety_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-root-cause", safety_pairs
        ).sem_map(
            cols=[
                {
                    "name": "electrical_root_cause",
                    "type": str,
                    "desc": (
                        "The electrical root-cause phrase shared by the complaint "
                        "and recall, in at most eight words."
                    ),
                }
            ],
            desc="Extract the shared electrical root-cause phrase.",
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            safety_pairs,
            ["electrical_root_cause"],
        )
        extracted["electrical_root_cause"] = extracted[
            "electrical_root_cause"
        ].map(normalize_root_cause)
        tracker.record_semantic(
            "sem_map",
            len(safety_pairs),
            extracted,
            extraction_result,
            time.time() - started,
        )

        aggregation_input = extracted.copy()
        aggregation_input["_complaint_date"] = pd.to_datetime(
            aggregation_input["received_date"],
            errors="coerce",
        )
        aggregation_input["_recall_date"] = pd.to_datetime(
            aggregation_input["campaign_report_received_date"],
            errors="coerce",
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
            grouped = (
                aggregation_input.groupby(
                    "campaign_number",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    matched_complaint_count=("complaint_id", "nunique"),
                    median_days_complaint_to_recall=(
                        "_days_to_recall",
                        rounded_median,
                    ),
                    share_complaints_before_recall=(
                        "_before_recall",
                        "mean",
                    ),
                    days_distribution=("_days_to_recall", collect_values),
                    root_causes=("electrical_root_cause", collect_values),
                    complaint_summaries=("summary_text", collect_values),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(aggregation_input), grouped)

        summary_rows = []
        for index, row in enumerate(
            grouped.to_dict(orient="records"),
            start=1,
        ):
            aggregate_input, aggregate_plan = build_campaign_summary_plan(
                index, row
            )
            started = time.time()
            aggregate_result = aggregate_plan.run(config)
            aggregate_frame = result_frame(aggregate_result)
            tracker.record_semantic(
                "sem_agg",
                len(aggregate_input),
                aggregate_frame,
                aggregate_result,
                time.time() - started,
            )
            if aggregate_frame.empty:
                raise ValueError(
                    f"{TASK_ID}: semantic aggregate returned no campaign summary"
                )
            parsed = parse_campaign_summary(
                aggregate_frame.iloc[0]["campaign_summary_json"]
            )
            summary_rows.append(
                {"campaign_number": row["campaign_number"], **parsed}
            )

        summaries = pd.DataFrame.from_records(
            summary_rows,
            columns=[
                "campaign_number",
                "response_promptness_label",
                "dominant_electrical_root_cause",
            ],
        )
        with_summaries = grouped.merge(
            summaries,
            on="campaign_number",
            how="inner",
        )
        result = with_summaries[[
            "campaign_number",
            "matched_complaint_count",
            "median_days_complaint_to_recall",
            "share_complaints_before_recall",
            "dominant_electrical_root_cause",
            "response_promptness_label",
        ]].copy()
        tracker.record("project", len(with_summaries), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
