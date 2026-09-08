#!/usr/bin/env python3
"""
vehicle_safety-038
Summarize Toyota OTHER-component recall campaigns using semantically matching
complaints, complaint sentiment, recall severity, and shared defect phrases.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_FILTER ->
     SEM_CLASSIFY(sentiment) -> SEM_EXTRACT(phrase, severity) ->
     GROUP_BY(campaign) -> PROJECT
Output: table, metric: f1
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

TASK_ID = "vehicle_safety-038"


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
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        toyota_other_complaints = complaints[
            (complaints["make"] == "TOYOTA")
            & (complaints["component_id"] == "OTHER")
        ]
        tracker.record(
            "FILTER(make='TOYOTA' AND component_id='OTHER')",
            len(complaints),
            len(toyota_other_complaints),
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

        toyota_other_recalls = recalls[
            (recalls["vehicle_make"] == "TOYOTA")
            & (recalls["component_id"] == "OTHER")
        ]
        tracker.record(
            "FILTER(vehicle.make='TOYOTA' AND component.component_id='OTHER')",
            len(recalls),
            len(toyota_other_recalls),
        )

        complaint_bindings = toyota_other_complaints[
            ["complaint_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=OTHER; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN complaint inputs)",
            len(toyota_other_complaints),
            len(complaint_bindings),
        )

        recall_bindings = toyota_other_recalls[
            ["campaign_number", "recall_defect_summary", "recall_consequence_summary"]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=OTHER; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN recall inputs)",
            len(toyota_other_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Toyota OTHER complaint and recall share defect)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            if complaint_bindings.empty or recall_bindings.empty:
                matched_pairs = pd.DataFrame(
                    columns=[
                        "complaint_id",
                        "summary_text",
                        "campaign_number",
                        "recall_defect_summary",
                        "recall_consequence_summary",
                    ]
                )
            else:
                matched_pairs = complaint_bindings.sem_join(
                    recall_bindings,
                    "Within Toyota OTHER records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same non-standard safety defect."
                )
            step.set_output(matched_pairs)

        with tracker.step(
            "SEM_FILTER(meaningful Toyota OTHER safety defect)",
            input_rows=len(matched_pairs),
        ) as step:
            if matched_pairs.empty:
                meaningful_pairs = matched_pairs.copy()
            else:
                meaningful_pairs = matched_pairs.sem_filter(
                    "The matched complaint {summary_text}, recall defect "
                    "{recall_defect_summary}, and consequence "
                    "{recall_consequence_summary} describe a meaningful safety "
                    "defect rather than a convenience, labeling, or cosmetic-only "
                    "concern."
                )
            step.set_output(meaningful_pairs)

        with tracker.step(
            "SEM_CLASSIFY(complaint_sentiment)",
            input_rows=len(meaningful_pairs),
        ) as step:
            if meaningful_pairs.empty:
                sentiment_labeled = meaningful_pairs.copy()
                sentiment_labeled["complaint_sentiment"] = pd.Series(dtype="object")
            else:
                sentiment_labeled = meaningful_pairs.sem_map(
                    "Assign complaint {summary_text} sentiment. Output exactly one "
                    "label: acute_safety_event, intermittent_issue, or "
                    "cosmetic_or_minor.",
                    suffix="complaint_sentiment",
                )
                sentiment_labeled["complaint_sentiment"] = (
                    sentiment_labeled["complaint_sentiment"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(sentiment_labeled)

        with tracker.step(
            "SEM_EXTRACT(shared_defect_phrase, recall_severity)",
            input_rows=len(sentiment_labeled),
        ) as step:
            if sentiment_labeled.empty:
                extracted = sentiment_labeled.copy()
                extracted["shared_defect_phrase"] = pd.Series(dtype="object")
                extracted["recall_severity"] = pd.Series(dtype="object")
            else:
                extracted = sentiment_labeled.sem_extract(
                    input_cols=[
                        "recall_defect_summary",
                        "recall_consequence_summary",
                    ],
                    output_cols={
                        "shared_defect_phrase": (
                            "A phrase of at most 8 words describing the defect "
                            "shared by the matched complaint and recall evidence."
                        ),
                        "recall_severity": (
                            "Exactly one recall severity label based on the defect "
                            "and consequence: critical, severe, moderate, or minor."
                        ),
                    },
                )
                extracted["shared_defect_phrase"] = (
                    extracted["shared_defect_phrase"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                )
                extracted["recall_severity"] = (
                    extracted["recall_severity"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(extracted)

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
            grouped = extracted.groupby("campaign_number", as_index=False).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                dominant_complaint_sentiment=(
                    "complaint_sentiment",
                    alpha_ascending_mode,
                ),
                dominant_recall_severity=(
                    "recall_severity",
                    alpha_ascending_mode,
                ),
                shared_defect_phrase=(
                    "shared_defect_phrase",
                    alpha_ascending_mode,
                ),
            )
        tracker.record(
            "GROUP_BY([campaign.number], matched count and alpha-tiebreak modes)",
            len(extracted),
            len(grouped),
        )

        result = grouped[
            [
                "campaign_number",
                "matched_complaint_count",
                "dominant_complaint_sentiment",
                "dominant_recall_severity",
                "shared_defect_phrase",
            ]
        ]
        tracker.record(
            "PROJECT([campaign_number, matched_complaint_count, dominant_complaint_sentiment, dominant_recall_severity, shared_defect_phrase])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
