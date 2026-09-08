#!/usr/bin/env python3
"""
vehicle_safety-004
For each distinct Kia recall campaign number in the OTHER component category
without a Park Outside consumer warning, what underlying failure mechanism
does its defect summary describe?
DAG: SCAN_DOCS(recalls) -> FILTER(vehicle.make='KIA' AND
     component.component_id='OTHER' AND risk.park_outside='FALSE')
     -> DEDUP([campaign.number]) -> SEM_EXTRACT(failure_mechanism, <=6 words)
Output: table, metric: table_f1
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_jsonl, save_output, df_records, StepTracker, Timer

TASK_ID = "vehicle_safety-004"

def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as t:
        rdf = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")
        tracker.record("SCAN_DOCS(recalls)", None, len(rdf))

        filtered = rdf[
            (rdf["vehicle_make"] == "KIA")
            & (rdf["component_component_id"] == "OTHER")
            & (rdf["risk_park_outside"] == "FALSE")
        ]
        tracker.record(
            "FILTER(vehicle.make='KIA' AND component.component_id='OTHER' AND risk.park_outside='FALSE')",
            len(rdf), len(filtered),
        )

        deduped = filtered.drop_duplicates(subset=["campaign_number"])
        tracker.record("DEDUP([campaign.number])", len(filtered), len(deduped))

        with tracker.step("SEM_EXTRACT(failure_mechanism)", input_rows=len(deduped)) as st:
            extracted = deduped.sem_extract(
                input_cols=["risk_defect_summary"],
                output_cols={
                    "failure_mechanism": (
                        "a short canonical phrase of at most six words describing "
                        "the underlying failure mechanism in the defect summary"
                    )
                },
            )
            st.set_output(extracted)

        result = extracted[["campaign_number", "failure_mechanism"]]
        tracker.record("PROJECT([campaign.number, failure_mechanism])",
                       len(extracted), len(result))
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
