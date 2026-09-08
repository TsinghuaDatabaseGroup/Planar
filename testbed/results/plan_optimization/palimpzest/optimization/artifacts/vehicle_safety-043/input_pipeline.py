#!/usr/bin/env python3
"""Plan-optimization pipeline for vehicle_safety-043."""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import palimpzest as pz  # noqa: E402
from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_jsonl,
    load_table,
    load_text_documents,
    memory_dataset,
    run_plan_optimization,
    save_output,
)

TASK_ID = "vehicle_safety-043"
DATASET = "nhtsa_vehicle_safety"
TARGET_MAKES = ("FORD", "JEEP", "HONDA", "TESLA")
RECALL_CORE_PATTERN = re.compile(
    r"(?<!\d)(\d{2})\s*[Vv]\s*-?\s*(\d{3})(?:\d{3})?(?!\d)"
)


def recall_core(value) -> str | None:
    match = RECALL_CORE_PATTERN.search(str(value))
    return None if match is None else f"{match.group(1)}V{match.group(2)}"


def add_recall_core(record: dict) -> dict:
    return {"recall_core": recall_core(record.get("campaign_number"))}


def normalize_recall_citations(record: dict) -> dict:
    raw_values = record.get("cited_recall_cores")
    values = (
        raw_values
        if isinstance(raw_values, (list, tuple, set))
        else [raw_values]
    )
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        if raw_value is None:
            continue
        for match in RECALL_CORE_PATTERN.finditer(str(raw_value)):
            core = f"{match.group(1)}V{match.group(2)}"
            if core not in seen:
                seen.add(core)
                normalized.append(core)
    return {"normalized_cited_recall_cores": normalized}


def add_join_key(_record: dict) -> dict:
    return {"join_key": 1}


def flatten_distinct_strings(value) -> list[str]:
    pending = list(value) if isinstance(value, (list, tuple, set)) else []
    flattened: list[str] = []
    while pending:
        item = pending.pop(0)
        if isinstance(item, (list, tuple, set)):
            pending[0:0] = list(item)
        elif item is not None:
            flattened.append(str(item))
    return sorted(set(flattened))


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        citations = memory_dataset(f"{TASK_ID}-reports", reports).sem_map(
            cols=[
                {
                    "name": "cited_recall_cores",
                    "type": list[str],
                    "desc": (
                        "Every cited NHTSA recall number normalized to its "
                        "two-digit year plus three-digit campaign core, with "
                        "duplicates removed."
                    ),
                }
            ],
            desc="Extract every NHTSA recall number cited in the report body.",
            depends_on=["body"],
        )
        citations = citations.map(
            normalize_recall_citations,
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
            add_join_key,
            cols=[
                {
                    "name": "join_key",
                    "type": int,
                    "desc": "Constant key for the global cross join.",
                }
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
                "vehicle_make": "make",
                "component_component_id": "component_id",
                "risk_defect_summary": "recall_defect_summary",
            }
        )
        selected_recalls = recalls.loc[
            recalls["make"].isin(TARGET_MAKES)
        ].reset_index(drop=True)
        recall_rows = memory_dataset(
            f"{TASK_ID}-recall-overview", selected_recalls
        ).map(
            add_recall_core,
            cols=[
                {
                    "name": "recall_core",
                    "type": str | None,
                    "desc": "Canonical recall campaign core.",
                }
            ],
            depends_on=["campaign_number"],
        )
        recall_by_make = recall_rows.groupby(
            pz.GroupBySig(
                group_by_fields=["make"],
                agg_funcs=["list", "list"],
                agg_fields=["campaign_number", "recall_core"],
            )
        )

        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "component_id", "summary_text"],
        )
        selected_complaints = complaints.loc[
            complaints["make"].isin(TARGET_MAKES)
        ].reset_index(drop=True)
        semantic_recalls = selected_recalls.rename(
            columns={
                "make": "recall_make",
                "component_id": "recall_component_id",
            }
        )[
            [
                "campaign_number",
                "recall_make",
                "recall_component_id",
                "recall_defect_summary",
            ]
        ]
        corroborated = memory_dataset(
            f"{TASK_ID}-complaints", selected_complaints
        ).sem_join(
            memory_dataset(f"{TASK_ID}-semantic-recalls", semantic_recalls),
            condition=(
                "Match only when make equals recall_make, component_id equals "
                "recall_component_id, and the complaint narrative describes "
                "the same underlying defect as the recall defect summary."
            ),
            depends_on=[
                "complaint_id",
                "make",
                "component_id",
                "summary_text",
                "campaign_number",
                "recall_make",
                "recall_component_id",
                "recall_defect_summary",
            ],
        )
        corroborated_by_make = corroborated.groupby(
            pz.GroupBySig(
                group_by_fields=["make"],
                agg_funcs=["list"],
                agg_fields=["complaint_id"],
            )
        )
        alignment = recall_by_make.join(
            corroborated_by_make,
            on="make",
            how="left",
        ).map(
            add_join_key,
            cols=[
                {
                    "name": "join_key",
                    "type": int,
                    "desc": "Constant key for the global cross join.",
                }
            ],
            depends_on=["make"],
        )
        plan = alignment.join(citation_summary, on="join_key", how="inner")

        started = time.time()
        optimized = run_plan_optimization(
            plan,
            config,
            task_id=TASK_ID,
            embedding_join_block_on=[
                ("make", "recall_make"),
                ("component_id", "recall_component_id"),
            ],
        )
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True)
        tracker.record_semantic(
            "optimized_plan",
            {
                "reports": len(reports),
                "recalls": len(selected_recalls),
                "complaints": len(selected_complaints),
            },
            output,
            optimized.result,
            time.time() - started,
        )

        answer = {}
        for row in output.to_dict(orient="records"):
            campaigns = flatten_distinct_strings(
                row.get("list(campaign_number)")
            )
            complaint_ids = flatten_distinct_strings(
                row.get("list(complaint_id)")
            )
            recall_cores = set(
                flatten_distinct_strings(row.get("list(recall_core)"))
            )
            cited_cores = set(
                flatten_distinct_strings(
                    row.get("list(normalized_cited_recall_cores)")
                )
            )
            answer[str(row["make"])] = {
                "corroborated_complaint_count": len(complaint_ids),
                "recall_campaign_count": len(campaigns),
                "alignment_label": (
                    "cited"
                    if recall_cores.intersection(cited_cores)
                    else "not_cited"
                ),
            }

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
