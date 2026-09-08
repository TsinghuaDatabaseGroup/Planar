#!/usr/bin/env python3
"""
vehicle_safety-043
Compare complaint, recall, and ODI investigation alignment for Ford, Jeep,
Honda, and Tesla.
DAG: report SEM_EXTRACT -> GROUP_BY([]); recall GROUP_BY(make) LEFT JOIN
     (complaint FILTER -> SEM_JOIN(recall FILTER) -> GROUP_BY(make)); CROSS JOIN
     report cores -> PROJECT(dictionary)
Output: dictionary, metric: f1
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (
    StepTracker,
    Timer,
    load_docs,
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-043"
TARGET_MAKES = ["FORD", "JEEP", "HONDA", "TESLA"]


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


def collect_recall_cores(values) -> list[str]:
    return sorted(
        {
            core
            for value in values.dropna()
            if (core := recall_core(value)) is not None
        }
    )


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

        recall_overview = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            ["campaign_number", "vehicle_make"]
        ].rename(columns={"vehicle_make": "make"})
        tracker.record("SCAN_DOCS(recalls)", None, len(recall_overview))

        selected_recall_overview = recall_overview[
            recall_overview["make"].isin(TARGET_MAKES)
        ]
        tracker.record(
            "FILTER(vehicle.make IN ['FORD','JEEP','HONDA','TESLA'])",
            len(recall_overview),
            len(selected_recall_overview),
        )

        recall_by_make = selected_recall_overview.groupby("make", as_index=False).agg(
            recall_campaign_count=("campaign_number", "nunique"),
            all_recall_cores=("campaign_number", collect_recall_cores),
        )
        tracker.record(
            "GROUP_BY([make], recall_campaign_count, all_recall_cores)",
            len(selected_recall_overview),
            len(recall_by_make),
        )

        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        selected_complaints = complaints[complaints["make"].isin(TARGET_MAKES)]
        tracker.record(
            "FILTER(make IN ['FORD','JEEP','HONDA','TESLA'])",
            len(complaints),
            len(selected_complaints),
        )

        semantic_recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_make": "recall_make",
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "recall_defect_summary",
            }
        )
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(semantic_recalls))

        selected_semantic_recalls = semantic_recalls[
            semantic_recalls["recall_make"].isin(TARGET_MAKES)
        ]
        tracker.record(
            "FILTER(r.make IN ['FORD','JEEP','HONDA','TESLA'])",
            len(semantic_recalls),
            len(selected_semantic_recalls),
        )

        complaint_bindings = selected_complaints[
            ["complaint_id", "make", "component_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_binding"] = (
            "make="
            + complaint_bindings["make"].fillna("").astype(str)
            + "; component_id="
            + complaint_bindings["component_id"].fillna("").astype(str)
            + "; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN complaint inputs)",
            len(selected_complaints),
            len(complaint_bindings),
        )

        recall_bindings = selected_semantic_recalls.copy()
        recall_bindings["recall_binding"] = (
            "make="
            + recall_bindings["recall_make"].fillna("").astype(str)
            + "; component_id="
            + recall_bindings["recall_component_id"].fillna("").astype(str)
            + "; recall_defect_summary="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN recall inputs)",
            len(selected_semantic_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(same make, component, and underlying defect)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            if complaint_bindings.empty or recall_bindings.empty:
                corroborated_pairs = pd.DataFrame(
                    columns=["complaint_id", "make"]
                )
            else:
                corroborated_pairs = complaint_bindings.sem_join(
                    recall_bindings,
                    "Match complaint {complaint_binding} to recall "
                    "{recall_binding} only when they have the same make and "
                    "component category and the complaint narrative describes the "
                    "same underlying defect as the recall defect summary."
                )
            step.set_output(corroborated_pairs)

        if corroborated_pairs.empty:
            corroborated_by_make = pd.DataFrame(
                columns=["make", "corroborated_complaint_count"]
            )
        else:
            corroborated_by_make = corroborated_pairs.groupby(
                "make", as_index=False
            ).agg(corroborated_complaint_count=("complaint_id", "nunique"))
        tracker.record(
            "GROUP_BY([make], corroborated_complaint_count=count_distinct(complaint_id))",
            len(corroborated_pairs),
            len(corroborated_by_make),
        )

        make_alignment = recall_by_make.merge(
            corroborated_by_make,
            on="make",
            how="left",
        )
        tracker.record(
            "JOIN(left, on=[make=right.make])",
            {
                "left": len(recall_by_make),
                "right": len(corroborated_by_make),
            },
            len(make_alignment),
        )

        aligned_with_citations = citation_summary.merge(make_alignment, how="cross")
        tracker.record(
            "JOIN(cross)",
            {"left": len(citation_summary), "right": len(make_alignment)},
            len(aligned_with_citations),
        )

        cited_cores = set(investigation_cited_cores)
        answer = {}
        for _, row in aligned_with_citations.iterrows():
            all_cores = set(row["all_recall_cores"])
            corroborated_count = row["corroborated_complaint_count"]
            answer[str(row["make"])] = {
                "corroborated_complaint_count": (
                    0 if pd.isna(corroborated_count) else int(corroborated_count)
                ),
                "recall_campaign_count": int(row["recall_campaign_count"]),
                "alignment_label": (
                    "cited" if all_cores.intersection(cited_cores) else "not_cited"
                ),
            }
        tracker.record(
            "PROJECT(format_as_dictionary, key=make, counts and alignment_label)",
            len(aligned_with_citations),
            1,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
