#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-043."""

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
    get_config,
    load_jsonl,
    load_table,
    load_text_documents,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-043"
DATASET = "nhtsa_vehicle_safety"
TARGET_MAKES = ("FORD", "JEEP", "HONDA", "TESLA")
RECALL_CORE_PATTERN = re.compile(
    r"(?<!\d)(\d{2})\s*[Vv]\s*-?\s*(\d{3})(?:\d{3})?(?!\d)"
)
MATCH_COLUMNS = [
    "complaint_id",
    "make",
    "component_id",
    "summary_text",
    "campaign_number",
    "recall_make",
    "recall_component_id",
    "recall_defect_summary",
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


def collect_recall_cores(values: pd.Series) -> list[str]:
    return sorted(
        {
            core
            for value in values.dropna()
            if (core := recall_core(value)) is not None
        }
    )


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
            desc="Extract every cited NHTSA recall number from the report body.",
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

        recall_overview = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "vehicle_make",
        ]].rename(columns={"vehicle_make": "make"})
        tracker.record("scan", None, recall_overview)

        selected_recall_overview = recall_overview.loc[
            recall_overview["make"].isin(TARGET_MAKES)
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(recall_overview),
            selected_recall_overview,
        )

        if selected_recall_overview.empty:
            recall_by_make = pd.DataFrame(
                columns=[
                    "make",
                    "recall_campaign_count",
                    "all_recall_cores",
                ]
            )
        else:
            recall_by_make = (
                selected_recall_overview.groupby(
                    "make",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    recall_campaign_count=("campaign_number", "nunique"),
                    all_recall_cores=("campaign_number", collect_recall_cores),
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(selected_recall_overview), recall_by_make)

        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "component_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        selected_complaints = complaints.loc[
            complaints["make"].isin(TARGET_MAKES),
            ["complaint_id", "make", "component_id", "summary_text"],
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), selected_complaints)

        semantic_recalls = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "vehicle_make",
            "component_component_id",
            "risk_defect_summary",
        ]].rename(
            columns={
                "vehicle_make": "recall_make",
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
            }
        )
        tracker.record("scan", None, semantic_recalls)

        selected_semantic_recalls = semantic_recalls.loc[
            semantic_recalls["recall_make"].isin(TARGET_MAKES),
            [
                "campaign_number",
                "recall_make",
                "recall_component_id",
                "recall_defect_summary",
            ],
        ].reset_index(drop=True)
        tracker.record("filter", len(semantic_recalls), selected_semantic_recalls)

        match_plan = memory_dataset(
            f"{TASK_ID}-complaints", selected_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-recalls", selected_semantic_recalls),
            condition=(
                "Within the same make and component category, match a complaint "
                "to a recall only when the complaint narrative describes the "
                "same underlying defect as the recall defect summary."
            ),
            depends_on=MATCH_COLUMNS,
        )
        started = time.time()
        match_result = match_plan.run(config)
        corroborated_pairs = result_frame(match_result)
        if corroborated_pairs.empty:
            corroborated_pairs = pd.DataFrame(columns=MATCH_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {
                "left": len(selected_complaints),
                "right": len(selected_semantic_recalls),
            },
            corroborated_pairs,
            match_result,
            time.time() - started,
        )

        if corroborated_pairs.empty:
            corroborated_by_make = pd.DataFrame(
                columns=["make", "corroborated_complaint_count"]
            )
        else:
            corroborated_by_make = (
                corroborated_pairs.groupby(
                    "make",
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    corroborated_complaint_count=(
                        "complaint_id",
                        "nunique",
                    )
                )
                .reset_index(drop=True)
            )
        tracker.record("groupby", len(corroborated_pairs), corroborated_by_make)

        make_alignment = recall_by_make.merge(
            corroborated_by_make,
            on="make",
            how="left",
        )
        tracker.record(
            "join",
            {"left": len(recall_by_make), "right": len(corroborated_by_make)},
            make_alignment,
        )

        aligned_with_citations = citation_summary.merge(
            make_alignment,
            how="cross",
        )
        tracker.record(
            "join",
            {"left": len(citation_summary), "right": len(make_alignment)},
            aligned_with_citations,
        )

        cited_cores = set(investigation_cited_cores)
        answer = {}
        for row in aligned_with_citations.to_dict(orient="records"):
            corroborated_count = row["corroborated_complaint_count"]
            answer[str(row["make"])] = {
                "corroborated_complaint_count": (
                    0
                    if pd.isna(corroborated_count)
                    else int(corroborated_count)
                ),
                "recall_campaign_count": int(row["recall_campaign_count"]),
                "alignment_label": (
                    "cited"
                    if set(row["all_recall_cores"]).intersection(cited_cores)
                    else "not_cited"
                ),
            }
        tracker.record("project", len(aligned_with_citations), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
