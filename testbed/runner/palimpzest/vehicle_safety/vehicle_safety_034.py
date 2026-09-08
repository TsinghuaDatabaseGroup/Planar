#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-034."""

from __future__ import annotations

import os
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
    normalize_enum,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-034"
DATASET = "nhtsa_vehicle_safety"
REPORT_COMPONENTS = (
    "adas",
    "airbag",
    "body",
    "brakes",
    "electrical",
    "engine",
    "fuel",
    "other",
    "powertrain",
    "seatbelt",
    "steering",
    "suspension",
    "tires",
)
SEMANTIC_JOIN_COLUMNS = [
    "complaint_id",
    "component_id",
    "summary_text",
    "campaign_number",
    "recall_component_id",
    "defect_summary",
]


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

        jeep_complaints = complaints.loc[
            complaints["make"] == "JEEP"
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), jeep_complaints)

        safety_plan = memory_dataset(
            f"{TASK_ID}-complaints", jeep_complaints
        ).sem_filter(
            filter=(
                "The complaint summary describes a genuine safety-relevant defect, "
                "not a cosmetic or convenience-only issue."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        safety_result = safety_plan.run(config)
        safety_complaints = result_frame(safety_result, jeep_complaints)
        tracker.record_semantic(
            "sem_filter",
            len(jeep_complaints),
            safety_complaints,
            safety_result,
            time.time() - started,
        )

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "component_component_id": "recall_component_id",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("scan", None, recalls)

        jeep_recalls = recalls.loc[
            recalls["vehicle_make"] == "JEEP",
            ["campaign_number", "recall_component_id", "defect_summary"],
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), jeep_recalls)

        left_records = safety_complaints[
            ["complaint_id", "component_id", "summary_text"]
        ].reset_index(drop=True)
        right_records = jeep_recalls[
            ["campaign_number", "recall_component_id", "defect_summary"]
        ].reset_index(drop=True)
        left_dataset = memory_dataset(f"{TASK_ID}-join-left", left_records)
        right_dataset = memory_dataset(f"{TASK_ID}-join-right", right_records)
        join_plan = left_dataset.sem_join(
            right_dataset,
            condition=(
                "Match the Jeep complaint to the Jeep recall only when their "
                "component categories are the same and the complaint narrative "
                "and recall defect summary describe the same underlying failure "
                "mechanism or a sufficiently specific defect condition. A broad "
                "shared symptom alone is not enough."
            ),
            depends_on=SEMANTIC_JOIN_COLUMNS,
        )
        started = time.time()
        join_result = join_plan.run(config)
        semantic_matches = result_frame(join_result)
        if semantic_matches.empty:
            semantic_matches = pd.DataFrame(columns=SEMANTIC_JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(left_records), "right": len(right_records)},
            semantic_matches,
            join_result,
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

        report_plan = memory_dataset(
            f"{TASK_ID}-reports", reports
        ).sem_map(
            cols=[
                {
                    "name": "covers_jeep",
                    "type": bool,
                    "desc": (
                        "True only when the report body explicitly identifies at "
                        "least one Jeep model in the investigated or reviewed "
                        "vehicle population. Do not infer coverage solely from an "
                        "unexplained recall number or manufacturer identity."
                    ),
                },
                {
                    "name": "report_component_id",
                    "type": str,
                    "desc": (
                        "Exactly one of ADAS, AIRBAG, BODY, BRAKES, ELECTRICAL, "
                        "ENGINE, FUEL, OTHER, POWERTRAIN, SEATBELT, STEERING, "
                        "SUSPENSION, or TIRES for the investigated safety issue."
                    ),
                },
            ],
            desc=(
                "Using only the report body, determine Jeep model coverage and "
                "map the investigated safety issue to one component category."
            ),
            depends_on=["body"],
        )
        started = time.time()
        report_result = report_plan.run(config)
        extracted_reports = result_frame(
            report_result,
            reports,
            ["covers_jeep", "report_component_id"],
        )
        extracted_reports["covers_jeep"] = extracted_reports[
            "covers_jeep"
        ].map(parse_bool)
        extracted_reports["report_component_id"] = extracted_reports[
            "report_component_id"
        ].map(lambda value: normalize_enum(value, REPORT_COMPONENTS))
        extracted_reports["report_component_id"] = extracted_reports[
            "report_component_id"
        ].str.upper()
        tracker.record_semantic(
            "sem_map",
            len(reports),
            extracted_reports,
            report_result,
            time.time() - started,
        )

        jeep_reports = extracted_reports.loc[
            extracted_reports["covers_jeep"]
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted_reports), jeep_reports)

        report_components = jeep_reports[
            ["report_component_id"]
        ].drop_duplicates().reset_index(drop=True)
        tracker.record("distinct", len(jeep_reports), report_components)

        matches_in_reported_components = semantic_matches.merge(
            report_components,
            left_on="component_id",
            right_on="report_component_id",
            how="inner",
        )
        tracker.record(
            "join",
            {"left": len(semantic_matches), "right": len(report_components)},
            matches_in_reported_components,
        )

        answer = int(matches_in_reported_components["complaint_id"].nunique())
        tracker.record("groupby", len(matches_in_reported_components), [answer])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
