#!/usr/bin/env python3
"""Plan-optimization pipeline for vehicle_safety-046."""

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

TASK_ID = "vehicle_safety-046"
DATASET = "nhtsa_vehicle_safety"
AIRBAG_FAILURE_MODES = (
    "non_deployment",
    "unintended_deployment",
    "inflator_rupture",
    "warning_or_sensor_fault",
    "other",
)
REMEDY_STYLES = ("software_update", "part_replacement", "other")


def normalize_failure_mode(record: dict) -> dict:
    return {
        "normalized_airbag_failure_mode": normalize_enum(
            record.get("airbag_failure_mode"), AIRBAG_FAILURE_MODES
        )
    }


def valid_failure_mode(record: dict) -> bool:
    return bool(record.get("normalized_airbag_failure_mode"))


def normalize_remedy_style(record: dict) -> dict:
    return {
        "normalized_remedy_style": normalize_enum(
            record.get("dominant_remedy_style"), REMEDY_STYLES
        )
    }


def valid_remedy_style(record: dict) -> bool:
    return bool(record.get("normalized_remedy_style"))


def distinct_strings(value) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else []
    return sorted({str(item) for item in values if item is not None})


def mode_from_list(value) -> str | None:
    values = value if isinstance(value, (list, tuple, set)) else []
    return stable_mode(pd.Series(list(values), dtype="object"))


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "component_id", "summary_text"],
        )
        chrysler_complaints = complaints.loc[
            (complaints["make"] == "CHRYSLER")
            & (complaints["component_id"] == "AIRBAG"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "vehicle_make",
            "component_component_id",
            "risk_defect_summary",
            "risk_consequence_summary",
            "remedy_corrective_action",
        ]].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
                "risk_consequence_summary": "recall_consequence_summary",
                "remedy_corrective_action": "recall_corrective_action",
            }
        )
        chrysler_recalls = recalls.loc[
            (recalls["vehicle_make"] == "CHRYSLER")
            & (recalls["recall_component_id"] == "AIRBAG"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
                "recall_corrective_action",
            ],
        ].reset_index(drop=True)

        matched = memory_dataset(
            f"{TASK_ID}-complaints", chrysler_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", chrysler_recalls),
            condition=(
                "Within Chrysler AIRBAG records, match a complaint to a recall "
                "only when component_id equals recall_component_id and the "
                "complaint narrative and recall defect summary describe the "
                "same airbag defect."
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
                "Keep this pair only if the matched evidence describes an "
                "airbag safety risk such as non-deployment, unintended "
                "deployment, inflator rupture, or occupant injury risk."
            ),
            depends_on=[
                "summary_text",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        )
        matched = matched.sem_map(
            cols=[
                {
                    "name": "airbag_failure_mode",
                    "type": str,
                    "desc": (
                        "Exactly one of non_deployment, unintended_deployment, "
                        "inflator_rupture, warning_or_sensor_fault, or other."
                    ),
                }
            ],
            desc="Assign the dominant airbag failure mode.",
            depends_on=["summary_text"],
        )
        matched = matched.map(
            normalize_failure_mode,
            cols=[
                {
                    "name": "normalized_airbag_failure_mode",
                    "type": str | None,
                    "desc": "Validated dominant airbag failure mode.",
                }
            ],
            depends_on=["airbag_failure_mode"],
        ).filter(
            valid_failure_mode,
            depends_on=["normalized_airbag_failure_mode"],
        )

        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        candidate_reports = reports.loc[
            reports["body"].str.contains(
                r"Chrysler|airbag|inflator",
                case=False,
                na=False,
                regex=True,
            ),
            ["file_id", "body"],
        ].reset_index(drop=True)
        report_matches = matched.sem_join(
            memory_dataset(f"{TASK_ID}-reports", candidate_reports),
            condition=(
                "Match the Chrysler airbag issue to an investigation report "
                "only when the report discusses the same airbag risk or ODI "
                "evidence."
            ),
            depends_on=[
                "normalized_airbag_failure_mode",
                "summary_text",
                "recall_defect_summary",
                "file_id",
                "body",
            ],
        )
        grouped = report_matches.groupby(
            pz.GroupBySig(
                group_by_fields=["campaign_number"],
                agg_funcs=["list", "list", "list", "list", "list"],
                agg_fields=[
                    "complaint_id",
                    "normalized_airbag_failure_mode",
                    "recall_corrective_action",
                    "file_id",
                    "body",
                ],
            )
        )
        summarized = grouped.sem_map(
            cols=[
                {
                    "name": "dominant_remedy_style",
                    "type": str,
                    "desc": (
                        "Exactly one of software_update, part_replacement, or "
                        "other."
                    ),
                },
                {
                    "name": "odi_evidence_summary",
                    "type": str,
                    "desc": "One short phrase summarizing the ODI evidence.",
                },
            ],
            desc=(
                "Assign the dominant remedy style from the collected corrective "
                "actions and summarize the related ODI evidence in one short "
                "phrase."
            ),
            depends_on=[
                "campaign_number",
                "list(recall_corrective_action)",
                "list(body)",
            ],
        )
        plan = summarized.map(
            normalize_remedy_style,
            cols=[
                {
                    "name": "normalized_remedy_style",
                    "type": str | None,
                    "desc": "Validated dominant recall remedy style.",
                }
            ],
            depends_on=["dominant_remedy_style"],
        ).filter(
            valid_remedy_style,
            depends_on=["normalized_remedy_style"],
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True).rename(
            columns={
                "list(complaint_id)": "complaint_ids",
                "list(normalized_airbag_failure_mode)": "failure_modes",
                "list(file_id)": "report_ids",
            }
        )
        output["dominant_remedy_style"] = output["normalized_remedy_style"]
        output["matched_complaint_count"] = output["complaint_ids"].map(
            lambda values: len(distinct_strings(values))
        )
        output["dominant_airbag_failure_mode"] = output[
            "failure_modes"
        ].map(mode_from_list)
        output["related_report_ids"] = output["report_ids"].map(
            distinct_strings
        )
        tracker.record_semantic(
            "optimized_plan",
            {
                "complaints": len(chrysler_complaints),
                "recalls": len(chrysler_recalls),
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
                    "dominant_airbag_failure_mode",
                    "dominant_remedy_style",
                    "related_report_ids",
                    "odi_evidence_summary",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
