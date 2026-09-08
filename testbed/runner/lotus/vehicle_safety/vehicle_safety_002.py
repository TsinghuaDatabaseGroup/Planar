#!/usr/bin/env python3
"""
vehicle_safety-002
Looking at every recall in the seatbelt component category, is it true that
each one describes a defect that could prevent the belt from restraining an
occupant during a crash and that none describes a purely cosmetic or
comfort-only issue?
DAG: SCAN_DOCS(recalls) -> FILTER(component.component_id='SEATBELT')
     -> SEM_AGGREGATE(all restrain-defect AND none cosmetic -> Yes/No)
Output: label (Yes/No), metric: label_accuracy
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline_helpers import setup, load_jsonl, save_output, clean_label, StepTracker, Timer

TASK_ID = "vehicle_safety-002"

def main():
    setup(max_tokens=512)
    tracker = StepTracker()

    with Timer() as t:
        rdf = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")
        tracker.record("SCAN_DOCS(recalls)", None, len(rdf))

        filtered = rdf[rdf["component_component_id"] == "SEATBELT"]
        tracker.record("FILTER(component.component_id='SEATBELT')", len(rdf), len(filtered))

        with tracker.step("SEM_AGGREGATE(all restrain-defect AND none cosmetic -> Yes/No)",
                          input_rows=len(filtered)) as st:
            agg = filtered.sem_agg(
                "Do all input {risk_defect_summary} texts describe a defect that "
                "could prevent the belt from restraining an occupant during a "
                "crash AND do none of them describe a purely cosmetic or "
                "comfort-only issue? Output exactly one word: Yes or No."
            )
            st.set_output(agg)

        answer = clean_label(agg["_output"].iloc[0])

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=t.elapsed, tracker=tracker)

if __name__ == "__main__":
    main()
