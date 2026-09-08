#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-028."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    get_config,
    load_jsonl,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-028"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["complaint_id", "make", "vehicle_id", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        selected_complaints = complaints.loc[
            complaints["make"].isin(["TOYOTA", "HONDA"])
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), selected_complaints)

        complaint_plan = memory_dataset(
            f"{TASK_ID}-complaints", selected_complaints
        ).sem_filter(
            filter=(
                "The complaint summary describes a severe safety event that "
                "happened while the vehicle was being driven, including a crash, "
                "injury, death, fire, airbag non-deployment, loss of control, brake "
                "failure, loss of power, or towing after a driving incident."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        complaint_result = complaint_plan.run(config)
        severe_complaints = result_frame(
            complaint_result,
            selected_complaints,
        )
        tracker.record_semantic(
            "sem_filter",
            len(selected_complaints),
            severe_complaints,
            complaint_result,
            time.time() - started,
        )

        recalls = load_jsonl(DATASET, "recalls.jsonl")[
            [
                "campaign_number",
                "vehicle_vehicle_id",
                "vehicle_make",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "vehicle_vehicle_id": "vehicle_id",
                "vehicle_make": "make",
                "risk_defect_summary": "defect_summary",
            }
        )
        tracker.record("scan", None, recalls)

        selected_recalls = recalls.loc[
            recalls["make"].isin(["TOYOTA", "HONDA"])
        ].reset_index(drop=True)
        tracker.record("filter", len(recalls), selected_recalls)

        recall_plan = memory_dataset(
            f"{TASK_ID}-recalls", selected_recalls
        ).sem_filter(
            filter=(
                "The recall defect summary describes a safety-critical defect "
                "that can cause a crash, injury, fire, loss of control, airbag "
                "failure, braking failure, or loss of motive power."
            ),
            depends_on=["defect_summary"],
        )
        started = time.time()
        recall_result = recall_plan.run(config)
        critical_recalls = result_frame(recall_result, selected_recalls)
        tracker.record_semantic(
            "sem_filter",
            len(selected_recalls),
            critical_recalls,
            recall_result,
            time.time() - started,
        )

        deduplicated = critical_recalls.drop_duplicates(
            subset=["campaign_number", "vehicle_id"]
        ).reset_index(drop=True)
        tracker.record("distinct", len(critical_recalls), deduplicated)

        joined = severe_complaints.merge(
            deduplicated[["vehicle_id"]],
            on="vehicle_id",
            how="inner",
        )
        tracker.record(
            "join",
            {"left": len(severe_complaints), "right": len(deduplicated)},
            joined,
        )

        grouped = (
            joined.groupby("make", as_index=False, dropna=False)
            .agg(matched_complaint_count=("complaint_id", "nunique"))
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(joined), grouped)

        counts = dict(
            zip(
                grouped["make"],
                grouped["matched_complaint_count"],
                strict=True,
            )
        )
        toyota_count = int(counts.get("TOYOTA", 0))
        honda_count = int(counts.get("HONDA", 0))
        if toyota_count > honda_count:
            higher_make = "TOYOTA"
        elif honda_count > toyota_count:
            higher_make = "HONDA"
        else:
            higher_make = "tie"

        answer = {
            "toyota_matched_complaint_count": toyota_count,
            "honda_matched_complaint_count": honda_count,
            "higher_matched_complaint_make": higher_make,
        }
        tracker.record("project", len(grouped), answer)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
