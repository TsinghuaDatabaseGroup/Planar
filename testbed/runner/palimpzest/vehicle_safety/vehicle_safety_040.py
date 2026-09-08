#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-040."""

from __future__ import annotations

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
    stable_mode,
)

TASK_ID = "vehicle_safety-040"
DATASET = "nhtsa_vehicle_safety"
COMPLAINT_SENTIMENTS = (
    "acute_safety_event",
    "intermittent_issue",
    "cosmetic_or_minor",
)
MATCH_COLUMNS = [
    "complaint_id",
    "component_id",
    "received_date",
    "summary_text",
    "campaign_number",
    "recall_component_id",
    "campaign_report_received_date",
    "recall_defect_summary",
]


def normalize_failure_mode(value) -> str:
    return " ".join(str(value).strip().removesuffix(".").lower().split())


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

        jeep_complaints = complaints.loc[
            (complaints["make"] == "JEEP")
            & (complaints["component_id"] == "ENGINE"),
            ["complaint_id", "component_id", "received_date", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), jeep_complaints)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "campaign_report_received_date",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
            }
        )
        tracker.record("scan", None, recalls)

        jeep_recalls = recalls.loc[
            (recalls["vehicle_make"] == "JEEP")
            & (recalls["recall_component_id"] == "ENGINE"),
            [
                "campaign_number",
                "campaign_report_received_date",
                "recall_component_id",
                "recall_defect_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), jeep_recalls)

        match_plan = memory_dataset(
            f"{TASK_ID}-complaints", jeep_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", jeep_recalls),
            condition=(
                "Within Jeep ENGINE records, match a complaint to a recall only "
                "when the complaint narrative and recall defect summary describe "
                "the same engine failure mechanism."
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
            {"left": len(jeep_complaints), "right": len(jeep_recalls)},
            matched_pairs,
            match_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-failure-mode", matched_pairs
        ).sem_map(
            cols=[
                {
                    "name": "failure_mode",
                    "type": str,
                    "desc": (
                        "A normalized engine failure mode in at most six words, "
                        "shared by the complaint and recall."
                    ),
                }
            ],
            desc="Extract the shared engine failure mode.",
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            matched_pairs,
            ["failure_mode"],
        )
        extracted["failure_mode"] = extracted["failure_mode"].map(
            normalize_failure_mode
        )
        tracker.record_semantic(
            "sem_map",
            len(matched_pairs),
            extracted,
            extraction_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety", extracted
        ).sem_filter(
            filter=(
                "The matched evidence describes an engine failure that can affect "
                "driving safety, such as stalling, loss of motive power, fire risk, "
                "or engine seizure."
            ),
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        safety_matches = result_frame(safety_result, extracted)
        tracker.record_semantic(
            "sem_filter",
            len(extracted),
            safety_matches,
            safety_result,
            time.time() - started,
        )

        sentiment_plan = memory_dataset(
            f"{TASK_ID}-sentiment", safety_matches
        ).sem_map(
            cols=[
                {
                    "name": "complaint_sentiment",
                    "type": str,
                    "desc": (
                        "Exactly one of acute_safety_event, intermittent_issue, "
                        "or cosmetic_or_minor."
                    ),
                }
            ],
            desc="Assign the complaint sentiment.",
            depends_on=["summary_text"],
        )
        started = time.time()
        sentiment_result = sentiment_plan.run(config)
        labeled = result_frame(
            sentiment_result,
            safety_matches,
            ["complaint_sentiment"],
        )
        labeled["complaint_sentiment"] = labeled[
            "complaint_sentiment"
        ].map(lambda value: normalize_enum(value, COMPLAINT_SENTIMENTS))
        if labeled["complaint_sentiment"].isna().any():
            raise ValueError(f"{TASK_ID}: invalid complaint sentiment label")
        tracker.record_semantic(
            "sem_map",
            len(safety_matches),
            labeled,
            sentiment_result,
            time.time() - started,
        )

        labeled = labeled.copy()
        labeled["_complaint_date"] = pd.to_datetime(
            labeled["received_date"],
            errors="coerce",
        )
        labeled["_recall_date"] = pd.to_datetime(
            labeled["campaign_report_received_date"],
            errors="coerce",
        )
        if labeled.empty:
            grouped = pd.DataFrame(
                columns=[
                    "failure_mode",
                    "matched_complaint_count",
                    "distinct_campaign_count",
                    "earliest_complaint_date",
                    "earliest_recall_date",
                    "pre_recall_complaint_count",
                    "dominant_complaint_sentiment",
                ]
            )
        else:
            grouped = (
                labeled.groupby(
                    "failure_mode",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    matched_complaint_count=("complaint_id", "nunique"),
                    distinct_campaign_count=("campaign_number", "nunique"),
                    earliest_complaint_date=("_complaint_date", "min"),
                    earliest_recall_date=("_recall_date", "min"),
                    dominant_complaint_sentiment=(
                        "complaint_sentiment",
                        stable_mode,
                    ),
                )
                .reset_index(drop=True)
            )
            pre_recall_counts = (
                labeled.loc[
                    labeled["_complaint_date"] < labeled["_recall_date"]
                ]
                .groupby("failure_mode")["complaint_id"]
                .nunique()
                .rename("pre_recall_complaint_count")
            )
            grouped = grouped.merge(
                pre_recall_counts,
                on="failure_mode",
                how="left",
            )
            grouped["pre_recall_complaint_count"] = grouped[
                "pre_recall_complaint_count"
            ].fillna(0).astype(int)
            grouped["earliest_complaint_date"] = grouped[
                "earliest_complaint_date"
            ].dt.strftime("%Y-%m-%d")
            grouped["earliest_recall_date"] = grouped[
                "earliest_recall_date"
            ].dt.strftime("%Y-%m-%d")
        tracker.record("groupby", len(labeled), grouped)

        result = grouped[
            [
                "failure_mode",
                "matched_complaint_count",
                "distinct_campaign_count",
                "earliest_complaint_date",
                "earliest_recall_date",
                "pre_recall_complaint_count",
                "dominant_complaint_sentiment",
            ]
        ].copy()
        tracker.record("project", len(grouped), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
