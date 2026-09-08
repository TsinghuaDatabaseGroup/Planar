#!/usr/bin/env python3
"""Plan-optimization pipeline for vehicle_safety-039."""

from __future__ import annotations

import os
import re
import sys
import time

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
    pz,
    result_frame,
    run_plan_optimization,
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
ANSWER_COLUMNS = [
    "rank",
    "campaign_number",
    "component_id",
    "corroborating_complaint_count",
    "cited_by_investigation",
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


def normalize_citations(record: dict) -> dict:
    return {
        "normalized_cited_recall_cores": normalize_recall_core_list(
            record.get("cited_recall_cores")
        )
    }


def summarize_citations(record: dict) -> dict:
    return {
        "investigation_cited_cores": sorted(
            {
                core
                for values in record["list(normalized_cited_recall_cores)"]
                for core in normalize_recall_core_list(values)
            }
        ),
        "join_key": 1,
    }


def campaign_summary(record: dict) -> dict:
    components = record["list(component_id)"]
    return {
        "component_id": components[0] if components else None,
        "corroborating_complaint_count": int(record["count(complaint_id)"]),
        "join_key": 1,
    }


def investigation_flag(record: dict) -> dict:
    cited_cores = set(record["investigation_cited_cores"])
    return {
        "cited_by_investigation": (
            "true"
            if recall_core(record["campaign_number"]) in cited_cores
            else "false"
        )
    }


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")

    with Timer() as timer:
        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        tracker.record("scan", None, reports)

        citations = memory_dataset(f"{TASK_ID}-reports", reports).sem_map(
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
        citations = citations.map(
            normalize_citations,
            cols=[
                {
                    "name": "normalized_cited_recall_cores",
                    "type": list[str],
                    "desc": "Validated canonical recall cores.",
                }
            ],
            depends_on=["cited_recall_cores"],
        )
        citation_summary = citations.groupby(
            pz.GroupBySig(
                group_by_fields=[],
                agg_funcs=["list"],
                agg_fields=["normalized_cited_recall_cores"],
            )
        ).map(
            summarize_citations,
            cols=[
                {
                    "name": "investigation_cited_cores",
                    "type": list[str],
                    "desc": "Distinct recall cores cited across all reports.",
                },
                {
                    "name": "join_key",
                    "type": int,
                    "desc": "Constant key for the global cross join.",
                },
            ],
            depends_on=["list(normalized_cited_recall_cores)"],
        )

        recalls = load_jsonl(DATASET, "recalls.jsonl")[[
            "campaign_number",
            "vehicle_make",
            "component_component_id",
            "risk_defect_summary",
        ]].rename(
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

        matches = memory_dataset(
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
        campaign_counts = matches.distinct(
            ["campaign_number", "complaint_id"]
        ).groupby(
            pz.GroupBySig(
                group_by_fields=["campaign_number"],
                agg_funcs=["list", "count"],
                agg_fields=["component_id", "complaint_id"],
            )
        ).map(
            campaign_summary,
            cols=[
                {
                    "name": "component_id",
                    "type": str | None,
                    "desc": "The component ID of the deduplicated campaign.",
                },
                {
                    "name": "corroborating_complaint_count",
                    "type": int,
                    "desc": "Number of distinct corroborating complaints.",
                },
                {
                    "name": "join_key",
                    "type": int,
                    "desc": "Constant key for the global cross join.",
                },
            ],
            depends_on=["list(component_id)", "count(complaint_id)"],
        )
        plan = citation_summary.join(
            campaign_counts, on="join_key", how="inner"
        ).map(
            investigation_flag,
            cols=[
                {
                    "name": "cited_by_investigation",
                    "type": str,
                    "desc": "Whether a report cites the campaign core: true or false.",
                }
            ],
            depends_on=["campaign_number", "investigation_cited_cores"],
        )

        started = time.time()
        optimized = run_plan_optimization(plan, config, task_id=TASK_ID)
        if optimized.result is None:
            return
        output = result_frame(
            optimized.result,
            deduplicated_recalls,
            ANSWER_COLUMNS[1:],
        )[ANSWER_COLUMNS[1:]].copy()
        tracker.record_semantic(
            "optimized_plan",
            {
                "reports": len(reports),
                "recalls": len(deduplicated_recalls),
                "complaints": len(ford_complaints),
            },
            output,
            optimized.result,
            time.time() - started,
        )

        ordered = output.sort_values(
            ["corroborating_complaint_count", "campaign_number"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(output), ordered)
        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_five)
        top_five.insert(0, "rank", range(1, len(top_five) + 1))
        result = top_five[ANSWER_COLUMNS].copy()
        tracker.record("project", len(top_five), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
