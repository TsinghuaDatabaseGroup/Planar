#!/usr/bin/env python3
"""Plan-optimization pipeline for vehicle_safety-045."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    run_plan_optimization,
    save_output,
)

TASK_ID = "vehicle_safety-045"
DATASET = "nhtsa_vehicle_safety"


def normalize_theme(record: dict) -> dict:
    raw_theme = record.get("defect_theme")
    normalized = " ".join(
        str(raw_theme or "").strip().removesuffix(".").lower().split()
    )
    return {"normalized_defect_theme": normalized or None}


def valid_theme(record: dict) -> bool:
    return bool(record.get("normalized_defect_theme"))


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
        chevrolet_complaints = complaints.loc[
            (complaints["make"] == "CHEVROLET")
            & (complaints["component_id"] == "BRAKES"),
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
        chevrolet_recalls = recalls.loc[
            (recalls["vehicle_make"] == "CHEVROLET")
            & (recalls["recall_component_id"] == "BRAKES"),
            [
                "campaign_number",
                "recall_component_id",
                "recall_defect_summary",
                "recall_consequence_summary",
            ],
        ].reset_index(drop=True)

        matched = memory_dataset(
            f"{TASK_ID}-complaints", chevrolet_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", chevrolet_recalls),
            condition=(
                "Within Chevrolet BRAKES records, match a complaint to a recall "
                "only when component_id equals recall_component_id and the "
                "complaint narrative and recall defect summary describe the "
                "same braking defect or risk."
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
        matched = matched.sem_map(
            cols=[
                {
                    "name": "defect_theme",
                    "type": str,
                    "desc": (
                        "The braking defect theme shared by the complaint and "
                        "recall, in no more than eight words."
                    ),
                }
            ],
            desc="Extract the shared braking defect theme.",
            depends_on=["summary_text", "recall_defect_summary"],
        )
        matched = matched.map(
            normalize_theme,
            cols=[
                {
                    "name": "normalized_defect_theme",
                    "type": str | None,
                    "desc": "Normalized non-empty braking defect theme.",
                }
            ],
            depends_on=["defect_theme"],
        ).filter(valid_theme, depends_on=["normalized_defect_theme"])
        matched = matched.sem_filter(
            (
                "Keep this pair only if the matched evidence describes a "
                "braking safety risk with a potential crash, loss-of-control, "
                "or extended-stopping consequence."
            ),
            depends_on=["summary_text", "recall_consequence_summary"],
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
                r"Chevrolet|GM|brake",
                case=False,
                na=False,
                regex=True,
            ),
            ["file_id", "body"],
        ].reset_index(drop=True)
        report_matches = matched.sem_join(
            memory_dataset(f"{TASK_ID}-reports", candidate_reports),
            condition=(
                "Match the braking defect theme to an investigation report "
                "only when the report discusses the same Chevrolet or GM "
                "braking issue or ODI risk finding."
            ),
            depends_on=[
                "normalized_defect_theme",
                "summary_text",
                "recall_defect_summary",
                "file_id",
                "body",
            ],
        )
        grouped = report_matches.groupby(
            pz.GroupBySig(
                group_by_fields=["normalized_defect_theme"],
                agg_funcs=["list", "list", "list", "list"],
                agg_fields=[
                    "complaint_id",
                    "campaign_number",
                    "file_id",
                    "body",
                ],
            )
        )
        plan = grouped.sem_map(
            cols=[
                {
                    "name": "odi_risk_summary",
                    "type": str,
                    "desc": (
                        "One short phrase summarizing the ODI braking risk "
                        "finding for this defect theme."
                    ),
                }
            ],
            desc=(
                "Summarize the ODI braking risk finding from the related "
                "investigation report bodies in one short phrase."
            ),
            depends_on=["normalized_defect_theme", "list(body)"],
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True).rename(
            columns={
                "normalized_defect_theme": "defect_theme",
                "list(complaint_id)": "complaint_ids",
                "list(campaign_number)": "campaign_numbers",
                "list(file_id)": "report_ids",
            }
        )
        output["matched_complaint_count"] = output["complaint_ids"].map(
            lambda values: len(distinct_strings(values))
        )
        output["distinct_campaign_count"] = output["campaign_numbers"].map(
            lambda values: len(distinct_strings(values))
        )
        output["related_report_ids"] = output["report_ids"].map(
            distinct_strings
        )
        tracker.record_semantic(
            "optimized_plan",
            {
                "complaints": len(chevrolet_complaints),
                "recalls": len(chevrolet_recalls),
                "reports": len(candidate_reports),
            },
            output,
            optimized.result,
            time.time() - started,
        )
        ordered = output.sort_values(
            ["matched_complaint_count", "defect_theme"],
            ascending=[False, True],
            kind="stable",
        ).head(5).reset_index(drop=True)
        ordered.insert(0, "rank", range(1, len(ordered) + 1))
        answer = df_records(
            ordered[
                [
                    "rank",
                    "defect_theme",
                    "matched_complaint_count",
                    "distinct_campaign_count",
                    "related_report_ids",
                    "odi_risk_summary",
                ]
            ]
        )

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
