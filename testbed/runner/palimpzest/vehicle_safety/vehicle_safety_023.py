#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-023."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-023"
DATASET = "nhtsa_vehicle_safety"


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        complaints = load_table(
            DATASET,
            "complaints.csv",
            ["state_of_incident", "medical_attention_flag", "summary_text"],
        )
        tracker.record("scan", None, complaints)

        states = complaints["state_of_incident"]
        recorded = complaints.loc[
            complaints["medical_attention_flag"]
            & states.notna()
            & states.astype(str).str.strip().ne("")
        ].reset_index(drop=True)
        tracker.record("filter", len(complaints), recorded)

        relation_plan = memory_dataset(TASK_ID, recorded).sem_filter(
            filter=(
                "The complaint summary is airbag-related, including airbags, "
                "inflators, pretensioners, or supplemental restraint system behavior."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        relation_result = relation_plan.run(config)
        airbag_related = result_frame(relation_result, recorded)
        tracker.record_semantic(
            "sem_filter",
            len(recorded),
            airbag_related,
            relation_result,
            time.time() - started,
        )

        hazard_plan = memory_dataset(
            f"{TASK_ID}-hazard", airbag_related
        ).sem_filter(
            filter=(
                "The complaint summary describes an airbag deployment failure, "
                "unintended deployment, inflator rupture, or a similar deployment "
                "hazard."
            ),
            depends_on=["summary_text"],
        )
        started = time.time()
        hazard_result = hazard_plan.run(config)
        deployment_hazards = result_frame(hazard_result, airbag_related)
        tracker.record_semantic(
            "sem_filter",
            len(airbag_related),
            deployment_hazards,
            hazard_result,
            time.time() - started,
        )

        grouped = (
            deployment_hazards.groupby(
                "state_of_incident",
                as_index=False,
                dropna=False,
            )
            .size()
            .rename(
                columns={
                    "state_of_incident": "state",
                    "size": "complaint_count",
                }
            )
        )
        tracker.record("groupby", len(deployment_hazards), grouped)

        ordered = grouped.sort_values(
            ["complaint_count", "state"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record("orderby", len(grouped), ordered)

        top_five = ordered.head(5).reset_index(drop=True)
        tracker.record("limit", len(ordered), top_five)
        answer = df_records(top_five[["state", "complaint_count"]])

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
