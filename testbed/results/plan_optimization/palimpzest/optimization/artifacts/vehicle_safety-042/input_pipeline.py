#!/usr/bin/env python3
"""Plan-optimization pipeline for vehicle_safety-042."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

import palimpzest as pz  # noqa: E402
from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_jsonl,
    load_table,
    load_text_documents,
    memory_dataset,
    normalize_enum,
    run_plan_optimization,
    save_output,
    stable_mode,
)

TASK_ID = "vehicle_safety-042"
DATASET = "nhtsa_vehicle_safety"
ADAS_TOPICS = (
    "autopilot_misuse",
    "unexpected_braking",
    "steering_control",
    "warning_or_supervision_gap",
    "other",
)


def normalize_topic(record: dict) -> dict:
    return {
        "normalized_adas_topic": normalize_enum(
            record.get("adas_topic"), ADAS_TOPICS
        )
    }


def valid_topic(record: dict) -> bool:
    return bool(record.get("normalized_adas_topic"))


def distinct_strings(value) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else []
    return sorted({str(item) for item in values if item is not None})


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "component_id", "summary_text"],
        )
        tesla_complaints = complaints.loc[
            (complaints["make"] == "TESLA")
            & (complaints["component_id"] == "ADAS"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "vehicle_make",
            "component_component_id",
            "risk_defect_summary",
            "risk_consequence_summary",
        ]].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
                "risk_consequence_summary": "recall_consequence_summary",
            }
        )
        tesla_recalls = recalls.loc[
            (recalls["vehicle_make"] == "TESLA")
            & (recalls["recall_component_id"] == "ADAS"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)

        matched = memory_dataset(
            f"{TASK_ID}-complaints", tesla_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", tesla_recalls),
            condition=(
                "Within Tesla ADAS records, match a complaint to a recall only "
                "when component_id equals recall_component_id and the complaint "
                "narrative and recall defect summary describe the same driver-"
                "assistance or automation-control problem."
            ),
            depends_on=[
                "complaint_id",
                "component_id",
                "summary_text",
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
            ],
        )
        matched = matched.sem_filter(
            (
                "Keep this pair only if the complaint describes a driving-"
                "context ADAS safety issue rather than general infotainment or "
                "account software behavior."
            ),
            depends_on=["summary_text"],
        )
        matched = matched.sem_map(
            cols=[
                {
                    "name": "adas_topic",
                    "type": str,
                    "desc": (
                        "Exactly one of autopilot_misuse, unexpected_braking, "
                        "steering_control, warning_or_supervision_gap, or other."
                    ),
                }
            ],
            desc="Assign the dominant Tesla ADAS defect topic.",
            depends_on=["summary_text"],
        )
        matched = matched.map(
            normalize_topic,
            cols=[
                {
                    "name": "normalized_adas_topic",
                    "type": str | None,
                    "desc": "Validated Tesla ADAS defect topic.",
                }
            ],
            depends_on=["adas_topic"],
        ).filter(valid_topic, depends_on=["normalized_adas_topic"])

        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        candidate_reports = reports.loc[
            reports["body"].str.contains(
                r"Tesla|Autopilot|driver assistance",
                case=False,
                na=False,
                regex=True,
            ),
            ["file_id", "body"],
        ].reset_index(drop=True)

        report_matches = matched.sem_join(
            memory_dataset(f"{TASK_ID}-reports", candidate_reports),
            condition=(
                "Match the Tesla ADAS complaint-recall issue to an "
                "investigation report only when the report discusses the same "
                "driver-assistance safety issue or ODI human-factors concern."
            ),
            depends_on=[
                "complaint_id",
                "campaign_number",
                "summary_text",
                "recall_defect_summary",
                "normalized_adas_topic",
                "file_id",
                "body",
            ],
        )
        plan = report_matches.groupby(
            pz.GroupBySig(
                group_by_fields=["campaign_number"],
                agg_funcs=["list", "list", "list"],
                agg_fields=[
                    "complaint_id",
                    "normalized_adas_topic",
                    "file_id",
                ],
            )
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True).rename(
            columns={
                "list(complaint_id)": "complaint_ids",
                "list(normalized_adas_topic)": "adas_topics",
                "list(file_id)": "report_ids",
            }
        )
        output["matched_complaint_count"] = output["complaint_ids"].map(
            lambda values: len(distinct_strings(values))
        )
        output["dominant_adas_topic"] = output["adas_topics"].map(
            lambda values: stable_mode(pd.Series(values, dtype="object"))
        )
        output["related_report_ids"] = output["report_ids"].map(
            distinct_strings
        )
        tracker.record_semantic(
            "optimized_plan",
            {
                "complaints": len(tesla_complaints),
                "recalls": len(tesla_recalls),
                "reports": len(candidate_reports),
            },
            output,
            optimized.result,
            time.time() - started,
        )
        answer = df_records(
            output[
                [
                    "campaign_number",
                    "matched_complaint_count",
                    "dominant_adas_topic",
                    "related_report_ids",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
