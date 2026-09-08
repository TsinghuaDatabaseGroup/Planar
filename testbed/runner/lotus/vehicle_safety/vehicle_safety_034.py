#!/usr/bin/env python3
"""
vehicle_safety-034
Count distinct genuine-safety Jeep complaints that semantically match a Jeep
recall and belong to a component covered by a Jeep investigation report.
DAG: complaint FILTER -> SEM_FILTER -> SEM_JOIN(recall FILTER) ->
     JOIN(report SEM_EXTRACT -> FILTER -> DEDUP) -> GROUP_BY([])
Output: scalar, metric: exact_match
"""
import os
import sys

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

TASK_ID = "vehicle_safety-034"


def as_boolean(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            ["complaint_id", "make", "component_id", "summary_text"]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        jeep_complaints = complaints[complaints["make"] == "JEEP"]
        tracker.record(
            "FILTER(make='JEEP')",
            len(complaints),
            len(jeep_complaints),
        )

        with tracker.step(
            "SEM_FILTER(genuine safety-relevant defect)",
            input_rows=len(jeep_complaints),
        ) as step:
            safety_complaints = jeep_complaints.sem_filter(
                "The complaint {summary_text} describes a genuine "
                "safety-relevant defect, not a cosmetic or convenience-only issue."
            )
            step.set_output(safety_complaints)

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
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
        tracker.record("SCAN_DOCS(recalls)", None, len(recalls))

        jeep_recalls = recalls[recalls["vehicle_make"] == "JEEP"]
        tracker.record(
            "FILTER(vehicle.make='JEEP')",
            len(recalls),
            len(jeep_recalls),
        )

        complaint_bindings = safety_complaints[
            ["complaint_id", "component_id", "summary_text"]
        ].copy()
        complaint_bindings["complaint_binding"] = (
            "component_id="
            + complaint_bindings["component_id"].fillna("").astype(str)
            + "; complaint_text="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN complaint inputs)",
            len(safety_complaints),
            len(complaint_bindings),
        )

        recall_bindings = jeep_recalls[
            ["campaign_number", "recall_component_id", "defect_summary"]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id="
            + recall_bindings["recall_component_id"].fillna("").astype(str)
            + "; defect_summary="
            + recall_bindings["defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN recall inputs)",
            len(jeep_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(same component and same underlying defect)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            if complaint_bindings.empty or recall_bindings.empty:
                matched_complaints = complaint_bindings.iloc[0:0].copy()
            else:
                matched_complaints = complaint_bindings.sem_join(
                    recall_bindings,
                    "Match the Jeep complaint {complaint_binding} to the Jeep "
                    "recall {recall_binding} only when they are in the same "
                    "component category and the complaint narrative and recall "
                    "defect summary describe the same underlying failure mechanism "
                    "or a sufficiently specific defect condition. A broad shared "
                    "symptom alone is not enough."
                )
            step.set_output(matched_complaints)

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
            "SEM_EXTRACT(covers_jeep, report_component_id)",
            input_rows=len(reports),
        ) as step:
            extracted_reports = reports.sem_extract(
                input_cols=["body"],
                output_cols={
                    "covers_jeep": (
                        "Boolean: whether the report body explicitly identifies at "
                        "least one Jeep model in the investigated or reviewed "
                        "vehicle population. Use only the report body; do not infer "
                        "Jeep coverage from an unexplained recall number, external "
                        "metadata, or manufacturer identity."
                    ),
                    "report_component_id": (
                        "Map the investigated safety issue to exactly one label: "
                        "ADAS, AIRBAG, BODY, BRAKES, ELECTRICAL, ENGINE, FUEL, "
                        "OTHER, POWERTRAIN, SEATBELT, STEERING, SUSPENSION, or TIRES."
                    ),
                },
            )
            extracted_reports["covers_jeep"] = extracted_reports[
                "covers_jeep"
            ].apply(as_boolean)
            extracted_reports["report_component_id"] = (
                extracted_reports["report_component_id"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.strip(".")
                .str.upper()
            )
            step.set_output(extracted_reports)

        jeep_reports = extracted_reports[extracted_reports["covers_jeep"]]
        tracker.record(
            "FILTER(covers_jeep=true)",
            len(extracted_reports),
            len(jeep_reports),
        )

        report_components = jeep_reports[["report_component_id"]].drop_duplicates()
        tracker.record(
            "DEDUP([report_component_id])",
            len(jeep_reports),
            len(report_components),
        )

        matched_in_reported_components = matched_complaints.merge(
            report_components,
            left_on="component_id",
            right_on="report_component_id",
            how="inner",
        )
        tracker.record(
            "JOIN(inner, on=[component_id=report_component_id])",
            {
                "left": len(matched_complaints),
                "right": len(report_components),
            },
            len(matched_in_reported_components),
        )

        answer = int(matched_in_reported_components["complaint_id"].nunique())
        tracker.record(
            "GROUP_BY([], n=count_distinct(complaint_id))",
            len(matched_in_reported_components),
            1,
        )

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
