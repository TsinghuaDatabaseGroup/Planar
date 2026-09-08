#!/usr/bin/env python3
"""
vehicle_safety-039
Rank Ford recall campaigns by semantically corroborating complaint count and
mark whether an ODI investigation report cites each campaign's recall core.
DAG: report SEM_EXTRACT -> GROUP_BY([]); recall FILTER -> DEDUP ->
     SEM_JOIN(complaint FILTER) -> GROUP_BY(campaign); CROSS JOIN -> PROJECT ->
     ORDER_BY -> LIMIT -> PROJECT(rank)
Output: ordered list, metric: ordered_f1
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (
    StepTracker,
    Timer,
    df_records,
    load_docs,
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-039"


def recall_core(value) -> str | None:
    compact = re.sub(r"[^0-9Vv]", "", str(value)).upper()
    match = re.search(r"(\d{2})V(\d{3})", compact)
    if match is None:
        return None
    return f"{match.group(1)}V{match.group(2)}"


def normalize_recall_core_list(value) -> list[str]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    normalized = []
    seen = set()
    for item in values:
        core = recall_core(item)
        if core is not None and core not in seen:
            seen.add(core)
            normalized.append(core)
    return normalized


def main():
    setup(max_tokens=512)
    tracker = StepTracker()

    with Timer() as timer:
        reports = load_docs(
            "nhtsa_vehicle_safety", "investigation_reports"
        ).rename(columns={"doc_id": "file_id", "contents": "body"})
        reports = reports[reports["file_id"].str.lower().str.endswith(".txt")]
        tracker.record(
            "SCAN_DOCS(investigation_reports, selector='*.txt')",
            None,
            len(reports),
        )

        with tracker.step(
            "SEM_EXTRACT(cited_recall_cores)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                extracted_reports = reports.copy()
                extracted_reports["cited_recall_cores"] = pd.Series(dtype="object")
            else:
                extracted_reports = reports.sem_extract(
                    input_cols=["body"],
                    output_cols={
                        "cited_recall_cores": (
                            "A JSON list of every NHTSA recall number cited in the "
                            "report body. Normalize each to the canonical core made "
                            "of the 2-digit year, V, and 3-digit campaign sequence "
                            "(for example 23V085), and deduplicate the list."
                        )
                    },
                )
                extracted_reports["cited_recall_cores"] = extracted_reports[
                    "cited_recall_cores"
                ].apply(normalize_recall_core_list)
            step.set_output(extracted_reports)

        investigation_cited_cores = sorted(
            {
                core
                for cores in extracted_reports.get(
                    "cited_recall_cores", pd.Series(dtype="object")
                )
                for core in normalize_recall_core_list(cores)
            }
        )
        citation_summary = pd.DataFrame(
            [{"investigation_cited_cores": investigation_cited_cores}]
        )
        tracker.record(
            "GROUP_BY([], investigation_cited_cores=collect_distinct(cited_recall_cores))",
            len(extracted_reports),
            len(citation_summary),
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
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
        tracker.record("SCAN_DOCS(recalls)", None, len(recalls))

        ford_recalls = recalls[recalls["vehicle_make"] == "FORD"]
        tracker.record(
            "FILTER(vehicle.make='FORD')",
            len(recalls),
            len(ford_recalls),
        )

        deduped_recalls = ford_recalls.drop_duplicates(subset=["campaign_number"])
        tracker.record(
            "DEDUP([campaign.number])",
            len(ford_recalls),
            len(deduped_recalls),
        )

        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints AS c)", None, len(complaints))

        ford_complaints = complaints[complaints["make"] == "FORD"]
        tracker.record(
            "FILTER(make='FORD')",
            len(complaints),
            len(ford_complaints),
        )

        recall_bindings = deduped_recalls[
            ["campaign_number", "component_id", "defect_summary"]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id="
            + recall_bindings["component_id"].fillna("").astype(str)
            + "; defect_summary="
            + recall_bindings["defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN recall inputs)",
            len(deduped_recalls),
            len(recall_bindings),
        )

        complaint_bindings = ford_complaints[
            ["complaint_id", "component_id", "summary_text"]
        ].rename(columns={"component_id": "complaint_component_id"})
        complaint_bindings = complaint_bindings.copy()
        complaint_bindings["complaint_binding"] = (
            "component_id="
            + complaint_bindings["complaint_component_id"].fillna("").astype(str)
            + "; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN complaint inputs)",
            len(ford_complaints),
            len(complaint_bindings),
        )

        with tracker.step(
            "SEM_JOIN(same component and same underlying Ford defect)",
            input_rows={
                "left": len(recall_bindings),
                "right": len(complaint_bindings),
            },
        ) as step:
            if recall_bindings.empty or complaint_bindings.empty:
                corroborating_pairs = pd.DataFrame(
                    columns=["campaign_number", "component_id", "complaint_id"]
                )
            else:
                corroborating_pairs = recall_bindings.sem_join(
                    complaint_bindings,
                    "Within the same component_id block, match Ford recall "
                    "{recall_binding} to Ford complaint {complaint_binding} only "
                    "when the complaint narrative describes the same underlying "
                    "defect as the recall defect summary."
                )
            step.set_output(corroborating_pairs)

        if corroborating_pairs.empty:
            campaign_counts = pd.DataFrame(
                columns=[
                    "campaign_number",
                    "component_id",
                    "corroborating_complaint_count",
                ]
            )
        else:
            campaign_counts = corroborating_pairs.groupby(
                "campaign_number", as_index=False
            ).agg(
                component_id=("component_id", "first"),
                corroborating_complaint_count=("complaint_id", "nunique"),
            )
        tracker.record(
            "GROUP_BY([campaign.number], component_id=any(), corroborating_complaint_count)",
            len(corroborating_pairs),
            len(campaign_counts),
        )

        cross_joined = citation_summary.merge(campaign_counts, how="cross")
        tracker.record(
            "JOIN(cross)",
            {"left": len(citation_summary), "right": len(campaign_counts)},
            len(cross_joined),
        )

        projected = cross_joined[
            ["campaign_number", "component_id", "corroborating_complaint_count"]
        ].copy()
        cited_cores = set(investigation_cited_cores)
        projected["cited_by_investigation"] = projected["campaign_number"].apply(
            lambda value: "true" if recall_core(value) in cited_cores else "false"
        )
        tracker.record(
            "PROJECT([campaign_number, component_id, corroborating_complaint_count, cited_by_investigation])",
            len(cross_joined),
            len(projected),
        )

        ordered = projected.sort_values(
            ["corroborating_complaint_count", "campaign_number"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([corroborating_complaint_count DESC, campaign_number ASC])",
            len(projected),
            len(ordered),
        )

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("LIMIT(5)", len(ordered), len(top_five))

        result = top_five.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result = result[
            [
                "rank",
                "campaign_number",
                "component_id",
                "corroborating_complaint_count",
                "cited_by_investigation",
            ]
        ]
        tracker.record(
            "PROJECT([rank, campaign_number, component_id, corroborating_complaint_count, cited_by_investigation])",
            len(top_five),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
