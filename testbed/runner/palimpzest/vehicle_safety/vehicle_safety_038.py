#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-038."""

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

TASK_ID = "vehicle_safety-038"
DATASET = "nhtsa_vehicle_safety"
COMPLAINT_SENTIMENTS = (
    "acute_safety_event",
    "intermittent_issue",
    "cosmetic_or_minor",
)
RECALL_SEVERITIES = ("critical", "severe", "moderate", "minor")
MATCH_COLUMNS = [
    "complaint_id",
    "component_id",
    "summary_text",
    "campaign_number",
    "recall_component_id",
    "recall_defect_summary",
    "recall_consequence_summary",
]


def normalize_phrase(value) -> str:
    return " ".join(str(value).strip().removesuffix(".").split())


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        toyota_complaints = complaints.loc[
            (complaints["make"] == "TOYOTA")
            & (complaints["component_id"] == "OTHER"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), toyota_complaints)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
                "risk_consequence_summary",
            ]
        ].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
                "risk_consequence_summary": "recall_consequence_summary",
            }
        )
        tracker.record("scan", None, recalls)

        toyota_recalls = recalls.loc[
            (recalls["vehicle_make"] == "TOYOTA")
            & (recalls["recall_component_id"] == "OTHER"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), toyota_recalls)

        match_plan = memory_dataset(
            f"{TASK_ID}-complaints", toyota_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", toyota_recalls),
            condition=(
                "Within Toyota OTHER records, match a complaint to a recall only "
                "when the complaint narrative and recall defect summary describe "
                "the same non-standard safety defect."
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
            {"left": len(toyota_complaints), "right": len(toyota_recalls)},
            matched_pairs,
            match_result,
            time.time() - started,
        )

        relevance_plan = memory_dataset(
            f"{TASK_ID}-meaningful", matched_pairs
        ).sem_filter(
            filter=(
                "The matched Toyota OTHER evidence describes a meaningful safety "
                "defect rather than a convenience, labeling, or cosmetic-only "
                "concern."
            ),
            depends_on=[
                "summary_text",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        meaningful_pairs = result_frame(relevance_result, matched_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(matched_pairs),
            meaningful_pairs,
            relevance_result,
            time.time() - started,
        )

        sentiment_plan = memory_dataset(
            f"{TASK_ID}-sentiment", meaningful_pairs
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
        sentiment_labeled = result_frame(
            sentiment_result,
            meaningful_pairs,
            ["complaint_sentiment"],
        )
        sentiment_labeled["complaint_sentiment"] = sentiment_labeled[
            "complaint_sentiment"
        ].map(lambda value: normalize_enum(value, COMPLAINT_SENTIMENTS))
        if sentiment_labeled["complaint_sentiment"].isna().any():
            raise ValueError(f"{TASK_ID}: invalid complaint sentiment label")
        tracker.record_semantic(
            "sem_map",
            len(meaningful_pairs),
            sentiment_labeled,
            sentiment_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", sentiment_labeled
        ).sem_map(
            cols=[
                {
                    "name": "shared_defect_phrase",
                    "type": str,
                    "desc": (
                        "A phrase of at most eight words describing the defect "
                        "shared by the complaint and recall evidence."
                    ),
                },
                {
                    "name": "recall_severity",
                    "type": str,
                    "desc": "Exactly one of critical, severe, moderate, or minor.",
                },
            ],
            desc=(
                "Extract the shared defect phrase and classify recall severity "
                "from the defect and consequence."
            ),
            depends_on=[
                "summary_text",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            sentiment_labeled,
            ["shared_defect_phrase", "recall_severity"],
        )
        extracted["shared_defect_phrase"] = extracted[
            "shared_defect_phrase"
        ].map(normalize_phrase)
        extracted["recall_severity"] = extracted["recall_severity"].map(
            lambda value: normalize_enum(value, RECALL_SEVERITIES)
        )
        if extracted["recall_severity"].isna().any():
            raise ValueError(f"{TASK_ID}: invalid recall severity label")
        tracker.record_semantic(
            "sem_map",
            len(sentiment_labeled),
            extracted,
            extraction_result,
            time.time() - started,
        )

        if extracted.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "dominant_complaint_sentiment",
                    "dominant_recall_severity",
                    "shared_defect_phrase",
                ]
            )
        else:
            grouped = (
                extracted.groupby(
                    "campaign_number",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    matched_complaint_count=("complaint_id", "nunique"),
                    dominant_complaint_sentiment=(
                        "complaint_sentiment",
                        stable_mode,
                    ),
                    dominant_recall_severity=(
                        "recall_severity",
                        stable_mode,
                    ),
                    shared_defect_phrase=(
                        "shared_defect_phrase",
                        stable_mode,
                    ),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(extracted), grouped)

        result = grouped[
            [
                "campaign_number",
                "matched_complaint_count",
                "dominant_complaint_sentiment",
                "dominant_recall_severity",
                "shared_defect_phrase",
            ]
        ].copy()
        tracker.record("project", len(grouped), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
