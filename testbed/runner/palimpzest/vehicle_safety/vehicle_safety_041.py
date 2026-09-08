#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-041."""

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
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-041"
DATASET = "nhtsa_vehicle_safety"
COMPLAINT_RECALL_COLUMNS = [
    "complaint_id",
    "component_id",
    "summary_text",
    "campaign_number",
    "recall_component_id",
    "recall_defect_summary",
    "recall_consequence_summary",
]
REPORT_MATCH_COLUMNS = [
    "complaint_id",
    "campaign_number",
    "summary_text",
    "recall_defect_summary",
    "defect_mechanism",
    "file_id",
    "body",
]


def distinct_sorted(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values.dropna()})


def distinct_in_order(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values.dropna()))


def normalize_mechanism(value) -> str:
    return " ".join(str(value).strip().removesuffix(".").lower().split())


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
    return {
        "dominant_defect_mechanism": str(
            parsed.get("dominant_defect_mechanism", "")
        ).strip(),
        "odi_risk_summary": str(parsed.get("odi_risk_summary", "")).strip(),
    }


def build_campaign_summary_plan(index: int, row: dict):
    aggregate_input = pd.DataFrame(
        [
            {
                "defect_mechanisms": row["defect_mechanisms"],
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
                "A JSON-encoded string whose decoded object has exactly "
                "the string fields dominant_defect_mechanism and "
                "odi_risk_summary."
            ),
        },
        agg=(
            "Identify the dominant defect mechanism and summarize the "
            "ODI risk signal for this campaign in one short phrase. "
            "Set the output field campaign_summary_json to a "
            "JSON-encoded string whose decoded object contains exactly "
            "the string fields dominant_defect_mechanism and "
            "odi_risk_summary. Do not place those two fields at the "
            "outer response level."
        ),
        depends_on=["defect_mechanisms", "report_bodies"],
    )
    return aggregate_input, aggregate_plan


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

        tesla_complaints = complaints.loc[
            (complaints["make"] == "TESLA")
            & (complaints["component_id"] == "ELECTRICAL"),
            ["complaint_id", "component_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), tesla_complaints)

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
        tracker.record("scan", None, recalls)

        tesla_recalls = recalls.loc[
            (recalls["vehicle_make"] == "TESLA")
            & (recalls["recall_component_id"] == "ELECTRICAL"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), tesla_recalls)

        pair_plan = memory_dataset(
            f"{TASK_ID}-complaints", tesla_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", tesla_recalls),
            condition=(
                "Within Tesla ELECTRICAL records, match a complaint to a recall "
                "only when the complaint narrative and recall defect summary "
                "describe the same electrical, warning, software-control, or "
                "power-loss defect."
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
            {"left": len(tesla_complaints), "right": len(tesla_recalls)},
            complaint_recall_pairs,
            pair_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-mechanism", complaint_recall_pairs
        ).sem_map(
            cols=[
                {
                    "name": "defect_mechanism",
                    "type": str,
                    "desc": (
                        "The dominant Tesla electrical defect mechanism shared "
                        "by the complaint and recall, in at most eight words."
                    ),
                }
            ],
            desc="Extract the dominant shared Tesla electrical defect mechanism.",
            depends_on=["summary_text", "recall_defect_summary"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            complaint_recall_pairs,
            ["defect_mechanism"],
        )
        extracted["defect_mechanism"] = extracted["defect_mechanism"].map(
            normalize_mechanism
        )
        tracker.record_semantic(
            "sem_map",
            len(complaint_recall_pairs),
            extracted,
            extraction_result,
            time.time() - started,
        )

        safety_plan = memory_dataset(
            f"{TASK_ID}-safety", extracted
        ).sem_filter(
            filter=(
                "The matched evidence describes a driving-safety or "
                "warning-control risk, not only an infotainment or convenience "
                "issue."
            ),
            depends_on=["summary_text", "recall_consequence_summary"],
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
                r"Tesla|Autopilot|electrical",
                case=False,
                na=False,
                regex=True,
            ),
            ["file_id", "body"],
        ].reset_index(drop=True)
        tracker.record("filter", len(reports), candidate_reports)

        report_left = safety_matches[[
            "complaint_id",
            "campaign_number",
            "summary_text",
            "recall_defect_summary",
            "defect_mechanism",
        ]].reset_index(drop=True)
        report_plan = memory_dataset(
            f"{TASK_ID}-report-left", report_left
        ).sem_join(
            memory_dataset(f"{TASK_ID}-report-right", candidate_reports),
            condition=(
                "Match the Tesla electrical defect mechanism to an investigation "
                "report only when the report discusses the same Tesla control, "
                "warning, or electrical-failure risk and ODI assessment."
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
                    "related_report_ids",
                    "defect_mechanisms",
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
                    related_report_ids=("file_id", distinct_sorted),
                    defect_mechanisms=("defect_mechanism", list),
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
                {
                    "campaign_number": row["campaign_number"],
                    **parsed,
                }
            )

        summaries = pd.DataFrame.from_records(
            summary_rows,
            columns=[
                "campaign_number",
                "dominant_defect_mechanism",
                "odi_risk_summary",
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
            "related_report_ids",
            "dominant_defect_mechanism",
            "odi_risk_summary",
        ]].copy()
        tracker.record("project", len(with_summaries), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
