#!/usr/bin/env python3
"""
vehicle_safety-008
For distinct Tesla recall campaigns in the steering component category that do
not include a Park Outside or Do Not Drive consumer warning, what primary
failure cause does each defect summary identify? Campaign number and a
canonical cause phrase of at most six words for each one.
DAG: SCAN_DOCS(recalls) -> FILTER(vehicle.make='TESLA' AND
     component.component_id='STEERING' AND risk.park_outside='FALSE' AND
     risk.do_not_drive='FALSE') -> DEDUP([campaign.number])
     -> SEM_EXTRACT(primary_cause_phrase, <=6 words)
Output: table, metric: table_f1
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_jsonl, save_output, df_records, StepTracker, Timer

TASK_ID = "vehicle_safety-008"

def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as t:
        rdf = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")
        tracker.record("SCAN_DOCS(recalls)", None, len(rdf))

        filtered = rdf[
            (rdf["vehicle_make"] == "TESLA")
            & (rdf["component_component_id"] == "STEERING")
            & (rdf["risk_park_outside"] == "FALSE")
            & (rdf["risk_do_not_drive"] == "FALSE")
        ]
        tracker.record(
            "FILTER(vehicle.make='TESLA' AND component.component_id='STEERING' "
            "AND risk.park_outside='FALSE' AND risk.do_not_drive='FALSE')",
            len(rdf), len(filtered),
        )

        deduped = filtered.drop_duplicates(subset=["campaign_number"])
        tracker.record("DEDUP([campaign.number])", len(filtered), len(deduped))

        with tracker.step("SEM_EXTRACT(primary_cause_phrase)", input_rows=len(deduped)) as st:
            extracted = deduped.sem_extract(
                input_cols=["risk_defect_summary"],
                output_cols={
                    "primary_cause_phrase": (
                        "the primary failure cause described in the defect summary, "
                        "as a short canonical phrase of at most six words"
                    )
                },
            )
            st.set_output(extracted)

        result = extracted[["campaign_number", "primary_cause_phrase"]]
        tracker.record("PROJECT([campaign.number, primary_cause_phrase])",
                       len(extracted), len(result))
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
