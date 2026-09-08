#!/usr/bin/env python3
"""
vehicle_safety-021
Match serious Nissan Altima airbag inflator complaints to airbag recalls by
vehicle ID and return the complaint, campaign, and extracted failure mode.
DAG: complaint branch FILTER -> SEM_FILTER -> SEM_EXTRACT; recall branch
     FILTER -> DEDUP; then JOIN(vehicle_id) -> PROJECT
Output: table, metric: table_f1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (
    StepTracker,
    Timer,
    df_records,
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-021"


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            [
                "complaint_id",
                "vehicle_id",
                "make",
                "model",
                "component_id",
                "injury_count",
                "death_count",
                "summary_text",
            ]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        serious_airbag = complaints[
            (complaints["make"] == "NISSAN")
            & (complaints["model"] == "ALTIMA")
            & (complaints["component_id"] == "AIRBAG")
            & (
                (complaints["injury_count"] > 0)
                | (complaints["death_count"] > 0)
            )
        ]
        tracker.record(
            "FILTER(make='NISSAN' AND model='ALTIMA' AND component_id='AIRBAG' AND (injury_count>0 OR death_count>0))",
            len(complaints),
            len(serious_airbag),
        )

        with tracker.step(
            "SEM_FILTER(inflator rupture, explosion, shrapnel, or metal fragments)",
            input_rows=len(serious_airbag),
        ) as step:
            rupture = serious_airbag.sem_filter(
                "The complaint {summary_text} describes an airbag inflator rupture, "
                "airbag explosion, shrapnel, or metal fragments causing injury or "
                "possible serious injury."
            )
            step.set_output(rupture)

        with tracker.step(
            "SEM_EXTRACT(airbag_failure_mode)", input_rows=len(rupture)
        ) as step:
            extracted = rupture.sem_extract(
                input_cols=["summary_text"],
                output_cols={
                    "airbag_failure_mode": (
                        "The primary airbag failure mode as a short canonical phrase."
                    )
                },
            )
            step.set_output(extracted)

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            ["campaign_number", "vehicle_vehicle_id", "component_component_id"]
        ].rename(columns={"vehicle_vehicle_id": "vehicle_id"})
        tracker.record("SCAN_DOCS(recalls)", None, len(recalls))

        airbag_recalls = recalls[
            recalls["component_component_id"] == "AIRBAG"
        ]
        tracker.record(
            "FILTER(component.component_id='AIRBAG')",
            len(recalls),
            len(airbag_recalls),
        )

        recall_keys = airbag_recalls[
            ["vehicle_id", "campaign_number"]
        ].drop_duplicates()
        tracker.record(
            "DEDUP([vehicle_id, campaign.number])",
            len(airbag_recalls),
            len(recall_keys),
        )

        joined = extracted.merge(recall_keys, on="vehicle_id", how="inner")
        tracker.record(
            "JOIN(inner, on=[vehicle_id])",
            {"left": len(extracted), "right": len(recall_keys)},
            len(joined),
        )

        result = joined[
            [
                "complaint_id",
                "vehicle_id",
                "campaign_number",
                "airbag_failure_mode",
            ]
        ]
        tracker.record(
            "PROJECT([complaint_id, vehicle_id, campaign_number, airbag_failure_mode])",
            len(joined),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
