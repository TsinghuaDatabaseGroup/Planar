#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-046."""

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
    load_text_documents,
    memory_dataset,
    normalize_enum,
    result_frame,
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
COMPLAINT_RECALL_COLUMNS = [
    "complaint_id",
    "component_id",
    "summary_text",
    "campaign_number",
    "recall_component_id",
    "recall_defect_summary",
    "recall_consequence_summary",
    "recall_corrective_action",
]
REPORT_MATCH_COLUMNS = [
    "complaint_id",
    "campaign_number",
    "summary_text",
    "recall_defect_summary",
    "airbag_failure_mode",
    "recall_corrective_action",
    "file_id",
    "body",
]


def distinct_sorted(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values.dropna()})


def distinct_in_order(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values.dropna()))


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
    remedy_style = normalize_enum(
        parsed.get("dominant_remedy_style", ""),
        REMEDY_STYLES,
    )
    if remedy_style is None:
        raise ValueError(f"{TASK_ID}: invalid dominant remedy style")
    return {
        "dominant_remedy_style": remedy_style,
        "odi_evidence_summary": str(
            parsed.get("odi_evidence_summary", "")
        ).strip(),
    }


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

        chrysler_complaints = complaints.loc[
            (complaints["make"] == "CHRYSLER")
            & (complaints["component_id"] == "AIRBAG"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), chrysler_complaints)

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
        tracker.record("scan", None, recalls)

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
        tracker.record("filter", len(recalls), chrysler_recalls)

        pair_plan = memory_dataset(
            f"{TASK_ID}-complaints", chrysler_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", chrysler_recalls),
            condition=(
                "Within Chrysler AIRBAG records, match a complaint to a recall "
                "only when the complaint narrative and recall defect summary "
                "describe the same airbag defect."
            ),
            depends_on=COMPLAINT_RECALL_COLUMNS,
        )
        started = time.time()
        pair_result = pair_plan.run(config)
        complaint_recall_pairs = result_frame(pair_result)
        if complaint_recall_pairs.empty:
            complaint_recall_pairs = pd.DataFrame(
                columns=COMPLAINT_RECALL_COLUMNS
            )
        tracker.record_semantic(
            "sem_join",
            {
                "left": len(chrysler_complaints),
                "right": len(chrysler_recalls),
            },
            complaint_recall_pairs,
            pair_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety", complaint_recall_pairs
        ).sem_filter(
            filter=(
                "The matched evidence describes an airbag safety risk such as "
                "non-deployment, unintended deployment, inflator rupture, or "
                "occupant injury risk."
            ),
            depends_on=[
                "summary_text",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        safety_pairs = result_frame(safety_result, complaint_recall_pairs)
        tracker.record_semantic(
            "sem_filter",
            len(complaint_recall_pairs),
            safety_pairs,
            safety_result,
            time.time() - started,
        )

        failure_plan = memory_dataset(
            f"{TASK_ID}-failure-mode", safety_pairs
        ).sem_map(
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
        started = time.time()
        failure_result = failure_plan.run(config)
        failure_labeled = result_frame(
            failure_result,
            safety_pairs,
            ["airbag_failure_mode"],
        )
        failure_labeled["airbag_failure_mode"] = failure_labeled[
            "airbag_failure_mode"
        ].map(lambda value: normalize_enum(value, AIRBAG_FAILURE_MODES))
        if failure_labeled["airbag_failure_mode"].isna().any():
            raise ValueError(f"{TASK_ID}: invalid airbag failure mode")
        tracker.record_semantic(
            "sem_map",
            len(safety_pairs),
            failure_labeled,
            failure_result,
            time.time() - started,
        )

        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        tracker.record("scan", None, reports)

        candidate_reports = reports.loc[
            reports["body"].str.contains(
                r"Chrysler|airbag|inflator",
                case=False,
                na=False,
                regex=True,
            ),
            ["file_id", "body"],
        ].reset_index(drop=True)
        tracker.record("filter", len(reports), candidate_reports)

        report_left = failure_labeled[[
            "complaint_id",
            "campaign_number",
            "summary_text",
            "recall_defect_summary",
            "airbag_failure_mode",
            "recall_corrective_action",
        ]].reset_index(drop=True)
        report_plan = memory_dataset(
            f"{TASK_ID}-report-left", report_left
        ).sem_join(
            memory_dataset(f"{TASK_ID}-report-right", candidate_reports),
            condition=(
                "Match the Chrysler airbag issue to an investigation report only "
                "when the report discusses the same airbag risk or ODI evidence."
            ),
            depends_on=REPORT_MATCH_COLUMNS,
        )
        started = time.time()
        report_result = report_plan.run(config)
        report_matches = result_frame(report_result)
        if report_matches.empty:
            report_matches = pd.DataFrame(columns=REPORT_MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(report_left), "right": len(candidate_reports)},
            report_matches,
            report_result,
            time.time() - started,
        )

        if report_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "matched_complaint_count",
                    "dominant_airbag_failure_mode",
                    "corrective_actions",
                    "related_report_ids",
                    "report_bodies",
                ]
            )
        else:
            grouped = (
                report_matches.groupby(
                    "campaign_number",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    matched_complaint_count=("complaint_id", "nunique"),
                    dominant_airbag_failure_mode=(
                        "airbag_failure_mode",
                        stable_mode,
                    ),
                    corrective_actions=(
                        "recall_corrective_action",
                        distinct_in_order,
                    ),
                    related_report_ids=("file_id", distinct_sorted),
                    report_bodies=("body", distinct_in_order),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(report_matches), grouped)

        summary_rows = []
        for index, row in enumerate(
            grouped.to_dict(orient="records"),
            start=1,
        ):
            aggregate_input = pd.DataFrame(
                [
                    {
                        "corrective_actions": row["corrective_actions"],
                        "report_bodies": row["report_bodies"],
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
                        "A JSON object with string fields dominant_remedy_style "
                        "and odi_evidence_summary."
                    ),
                },
                agg=(
                    "Assign dominant_remedy_style as exactly one of "
                    "software_update, part_replacement, or other from the "
                    "corrective actions, and summarize the ODI evidence from the "
                    "related reports in one short phrase. Return only a JSON "
                    "object with dominant_remedy_style and odi_evidence_summary."
                ),
                depends_on=["corrective_actions", "report_bodies"],
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
                "dominant_remedy_style",
                "odi_evidence_summary",
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
            "dominant_airbag_failure_mode",
            "dominant_remedy_style",
            "related_report_ids",
            "odi_evidence_summary",
        ]].copy()
        tracker.record("project", len(with_summaries), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
