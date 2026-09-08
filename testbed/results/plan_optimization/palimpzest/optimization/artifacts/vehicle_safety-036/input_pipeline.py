#!/usr/bin/env python3
"""Plan-optimization pipeline for vehicle_safety-036."""

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
    memory_dataset,
    normalize_enum,
    parse_bool,
    run_plan_optimization,
    save_output,
    stable_mode,
)

TASK_ID = "vehicle_safety-036"
DATASET = "nhtsa_vehicle_safety"
COMPLAINT_SEVERITIES = (
    "minor_inconvenience",
    "moderate_safety_event",
    "severe_safety_event",
)
REMEDY_STYLES = ("software_update", "part_replacement", "other")


def normalize_classifications(record: dict) -> dict:
    injury_count = pd.to_numeric(record.get("injury_count"), errors="coerce")
    return {
        "normalized_complaint_severity": (
            normalize_enum(record.get("complaint_severity"), COMPLAINT_SEVERITIES)
            or "minor_inconvenience"
        ),
        "normalized_recall_remedy_style": (
            normalize_enum(record.get("recall_remedy_style"), REMEDY_STYLES)
            or "other"
        ),
        "severe_indicator": bool(
            parse_bool(record.get("crash_flag"))
            or (not pd.isna(injury_count) and float(injury_count) > 0)
            or parse_bool(record.get("vehicle_towed_flag"))
        ),
    }


def list_mode(value) -> str | None:
    values = value if isinstance(value, list) else []
    return stable_mode(pd.Series(values, dtype="object")) if values else None


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            [
                "complaint_id",
                "make",
                "component_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "summary_text",
            ],
        )
        ford_complaints = complaints.loc[
            (complaints["make"] == "FORD")
            & (complaints["component_id"] == "AIRBAG"),
            [
                "complaint_id",
                "component_id",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
                "summary_text",
            ],
        ].reset_index(drop=True)
        recalls = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "vehicle_make",
            "component_component_id",
            "risk_defect_summary",
            "remedy_corrective_action",
        ]].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
                "remedy_corrective_action": "recall_corrective_action",
            }
        )
        ford_recalls = recalls.loc[
            (recalls["vehicle_make"] == "FORD")
            & (recalls["recall_component_id"] == "AIRBAG"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_corrective_action",
            ],
        ].reset_index(drop=True)

        plan = memory_dataset(f"{TASK_ID}-complaints", ford_complaints).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", ford_recalls),
            condition=(
                "Within Ford AIRBAG records, match a complaint to a recall only "
                "when component_id equals recall_component_id and the complaint "
                "narrative and recall defect summary describe the same airbag "
                "failure mechanism."
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
        plan = plan.sem_filter(
            (
                "Keep this matched pair only if the complaint narrative "
                "describes a concrete airbag safety event rather than a "
                "cosmetic warning-light-only concern."
            ),
            depends_on=["summary_text"],
        )
        plan = plan.sem_map(
            cols=[
                {
                    "name": "complaint_severity",
                    "type": str,
                    "desc": (
                        "Exactly one of minor_inconvenience, "
                        "moderate_safety_event, or severe_safety_event."
                    ),
                }
            ],
            desc="Classify the complaint severity.",
            depends_on=["summary_text"],
        )
        plan = plan.sem_map(
            cols=[
                {
                    "name": "recall_remedy_style",
                    "type": str,
                    "desc": (
                        "Exactly one of software_update, part_replacement, or "
                        "other."
                    ),
                }
            ],
            desc="Classify the recall remedy style.",
            depends_on=["recall_corrective_action"],
        )
        plan = plan.map(
            normalize_classifications,
            cols=[
                {
                    "name": "normalized_complaint_severity",
                    "type": str,
                    "desc": "Validated complaint severity label.",
                },
                {
                    "name": "normalized_recall_remedy_style",
                    "type": str,
                    "desc": "Validated recall remedy style.",
                },
                {
                    "name": "severe_indicator",
                    "type": bool,
                    "desc": "Whether structured fields indicate severe harm.",
                },
            ],
            depends_on=[
                "complaint_severity",
                "recall_remedy_style",
                "crash_flag",
                "injury_count",
                "vehicle_towed_flag",
            ],
        )
        plan = plan.distinct(
            [
                "campaign_number",
                "complaint_id",
                "normalized_complaint_severity",
                "normalized_recall_remedy_style",
                "severe_indicator",
            ]
        )
        plan = plan.groupby(
            pz.GroupBySig(
                group_by_fields=["campaign_number"],
                agg_funcs=["count", "average", "list", "list"],
                agg_fields=[
                    "complaint_id",
                    "severe_indicator",
                    "normalized_complaint_severity",
                    "normalized_recall_remedy_style",
                ],
            )
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True).rename(
            columns={
                "count(complaint_id)": "matched_complaint_count",
                "average(severe_indicator)": "severe_indicator_rate",
                "list(normalized_complaint_severity)": "severity_values",
                "list(normalized_recall_remedy_style)": "remedy_values",
            }
        )
        output["dominant_complaint_severity"] = output["severity_values"].map(
            list_mode
        )
        output["dominant_recall_remedy_style"] = output["remedy_values"].map(
            list_mode
        )
        tracker.record_semantic(
            "optimized_plan",
            {"complaints": len(ford_complaints), "recalls": len(ford_recalls)},
            output,
            optimized.result,
            time.time() - started,
        )
        ordered = output.sort_values(
            ["matched_complaint_count", "campaign_number"],
            ascending=[False, True],
            kind="stable",
        ).head(5).reset_index(drop=True)
        ordered.insert(0, "rank", range(1, len(ordered) + 1))
        answer = df_records(
            ordered[
                [
                    "rank",
                    "campaign_number",
                    "matched_complaint_count",
                    "severe_indicator_rate",
                    "dominant_complaint_severity",
                    "dominant_recall_remedy_style",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
