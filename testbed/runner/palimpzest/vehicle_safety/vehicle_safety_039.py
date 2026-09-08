#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-039."""

from __future__ import annotations

import os
import re
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

TASK_ID = "vehicle_safety-039"
DATASET = "nhtsa_vehicle_safety"
RECALL_CORE_PATTERN = re.compile(
    r"(?<!\d)(\d{2})\s*[Vv]\s*-?\s*(\d{3})(?:\d{3})?(?!\d)"
)
MATCH_COLUMNS = [
    "campaign_number",
    "component_id",
    "defect_summary",
    "complaint_id",
    "complaint_component_id",
    "summary_text",
]


def recall_core(value) -> str | None:
    match = RECALL_CORE_PATTERN.search(str(value))
    if match is None:
        return None
    return f"{match.group(1)}V{match.group(2)}"


def normalize_recall_core_list(value) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    normalized = []
    seen = set()
    for raw_value in values:
        if raw_value is None:
            continue
        for match in RECALL_CORE_PATTERN.finditer(str(raw_value)):
            core = f"{match.group(1)}V{match.group(2)}"
            if core not in seen:
                seen.add(core)
                normalized.append(core)
    return normalized


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        tracker.record("scan", None, reports)

        citation_plan = memory_dataset(
            f"{TASK_ID}-reports", reports
        ).sem_map(
            cols=[
                {
                    "name": "cited_recall_cores",
                    "type": list[str],
                    "desc": (
                        "Every NHTSA recall number cited in the report body, "
                        "normalized to the canonical two-digit year, V, and "
                        "three-digit campaign sequence; remove duplicates."
                    ),
                }
            ],
            desc="Extract all cited NHTSA recall numbers from the report body.",
            depends_on=["body"],
        )
        started = time.time()
        citation_result = citation_plan.run(config)
        extracted_reports = result_frame(
            citation_result,
            reports,
            ["cited_recall_cores"],
        )
        extracted_reports["cited_recall_cores"] = extracted_reports[
            "cited_recall_cores"
        ].map(normalize_recall_core_list)
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted_reports,
            citation_result,
            time.time() - started,
        )

        investigation_cited_cores = sorted(
            {
                core
                for cores in extracted_reports["cited_recall_cores"]
                for core in cores
            }
        )
        citation_summary = pd.DataFrame(
            [{"investigation_cited_cores": investigation_cited_cores}]
        )
        tracker.record("groupby", len(extracted_reports), citation_summary)

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "component_component_id": "component_id",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("scan", None, recalls)

        ford_recalls = recalls.loc[
            recalls["vehicle_make"] == "FORD",
            ["campaign_number", "component_id", "defect_summary"],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), ford_recalls)

        deduplicated_recalls = ford_recalls.drop_duplicates(
            subset=["campaign_number"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(ford_recalls), deduplicated_recalls)

        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        ford_complaints = complaints.loc[
            complaints["make"] == "FORD",
            ["complaint_id", "component_id", "summary_text"],
        ].rename(columns={"component_id": "complaint_component_id"})
        ford_complaints = ford_complaints.reset_index(drop=True)
        tracker.record("filter", len(complaints), ford_complaints)

        match_plan = memory_dataset(
            f"{TASK_ID}-recalls", deduplicated_recalls
        ).sem_join(
            memory_dataset(f"{TASK_ID}-complaints", ford_complaints),
            condition=(
                "Within the same component category, match a Ford recall campaign "
                "to a Ford complaint only when the complaint narrative describes "
                "the same underlying defect as the recall defect summary."
            ),
            depends_on=MATCH_COLUMNS,
        )
        started = time.time()
        match_result = match_plan.run(config)
        corroborating_pairs = result_frame(match_result)
        if corroborating_pairs.empty:
            corroborating_pairs = pd.DataFrame(columns=MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {
                "left": len(deduplicated_recalls),
                "right": len(ford_complaints),
            },
            corroborating_pairs,
            match_result,
            time.time() - started,
        )

        if corroborating_pairs.empty:
            campaign_counts = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "component_id",
                    "corroborating_complaint_count",
                ]
            )
        else:
            campaign_counts = (
                corroborating_pairs.groupby(
                    "campaign_number",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    component_id=("component_id", "first"),
                    corroborating_complaint_count=(
                        "complaint_id",
                        "nunique",
                    ),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(corroborating_pairs), campaign_counts)

        cross_joined = citation_summary.merge(campaign_counts, how="cross")
        tracker.record(
            "join",
            {"left": len(citation_summary), "right": len(campaign_counts)},
            cross_joined,
        )

        projected = cross_joined[
            ["campaign_number", "component_id", "corroborating_complaint_count"]
        ].copy()
        cited_cores = set(investigation_cited_cores)
        projected["cited_by_investigation"] = projected[
            "campaign_number"
        ].map(
            lambda value: "true" if recall_core(value) in cited_cores else "false"
        )
        tracker.record("project", len(cross_joined), projected)

        ordered = projected.sort_values(
            ["corroborating_complaint_count", "campaign_number"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(projected), ordered)

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_five)

        top_five.insert(0, "rank", range(1, len(top_five) + 1))
        result = top_five[
            [
                "rank",
                "campaign_number",
                "component_id",
                "corroborating_complaint_count",
                "cited_by_investigation",
            ]
        ].copy()
        tracker.record("project", len(top_five), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
