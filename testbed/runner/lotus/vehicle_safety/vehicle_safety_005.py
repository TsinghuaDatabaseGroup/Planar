#!/usr/bin/env python3
"""
vehicle_safety-005
Which Tesla recall campaign numbers have no Park Outside or Do Not Drive
consumer warning, use an over-the-air software update as the corrective
action, and describe a steering or motion-control issue in the defect summary?
DAG: SCAN_DOCS(recalls) -> FILTER(vehicle.make='TESLA' AND
     risk.park_outside='FALSE' AND risk.do_not_drive='FALSE')
     -> SEM_FILTER(OTA update AND steering/motion-control issue)
     -> PROJECT([campaign_number, model, model_year, defect_summary])
Output: table, metric: table_f1
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_jsonl, save_output, df_records, StepTracker, Timer

TASK_ID = "vehicle_safety-005"

def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as t:
        rdf = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")
        tracker.record("SCAN_DOCS(recalls)", None, len(rdf))

        filtered = rdf[
            (rdf["vehicle_make"] == "TESLA")
            & (rdf["risk_park_outside"] == "FALSE")
            & (rdf["risk_do_not_drive"] == "FALSE")
        ]
        tracker.record(
            "FILTER(vehicle.make='TESLA' AND risk.park_outside='FALSE' AND risk.do_not_drive='FALSE')",
            len(rdf), len(filtered),
        )

        with tracker.step("SEM_FILTER(OTA software update AND steering/motion-control issue)",
                          input_rows=len(filtered)) as st:
            hits = filtered.sem_filter(
                "The corrective action {remedy_corrective_action} describes an "
                "over-the-air software update AND the defect summary "
                "{risk_defect_summary} describes a steering or motion-control issue"
            )
            st.set_output(hits)

        result = hits[["campaign_number", "vehicle_model", "vehicle_model_year",
                       "risk_defect_summary"]].rename(columns={
            "vehicle_model": "model",
            "vehicle_model_year": "model_year",
            "risk_defect_summary": "defect_summary",
        })
        tracker.record("PROJECT([campaign_number, model, model_year, defect_summary])",
                       len(hits), len(result))
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
