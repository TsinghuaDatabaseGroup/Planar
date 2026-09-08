#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-022."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_jsonl,
    parse_number,
    parse_record_list,
    save_output,
    setup,
)


TASK_ID = "aviation_safety-022"


def main():
    setup(max_tokens=1536)
    tracker = StepTracker()

    with Timer() as timer:
        candidates = load_jsonl(
            "operator_implement",
            "inputs/NASA_ASRS-030_safety_barrier_candidates.jsonl",
        )[
            [
                "incident_id",
                "flight_phase",
                "aircraft_operator",
                "text",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(operator_implement/inputs/"
            "NASA_ASRS-030_safety_barrier_candidates.jsonl)",
            None,
            len(candidates),
            output=candidates,
        )

        with tracker.step(
            "SEM_AGGREGATE(top successful safety-barrier combinations)",
            input_rows=len(candidates),
        ) as step:
            aggregated = candidates.sem_agg(
                "Across all input records with incident IDs {incident_id}, "
                "flight phases {flight_phase}, aircraft operators "
                "{aircraft_operator}, and narratives {text}, count a successful "
                "barrier only when the narrative clearly establishes hazard "
                "detection, a concrete intervention, and avoidance of a worse "
                "consequence or stabilization. For every flight-phase and "
                "aircraft-operator combination, count distinct candidate "
                "incidents and distinct successful incidents. Keep combinations "
                "with at least 20 candidates, calculate successful_barrier_rate "
                "as successful count divided by candidate count, and return the "
                "top five ranked by rate descending, successful count descending, "
                "flight phase ascending, and aircraft operator ascending. Output "
                "only a valid JSON array with keys rank, flight_phase, "
                "aircraft_operator, candidate_incident_count, "
                "successful_barrier_count, and successful_barrier_rate, without "
                "markdown or commentary."
            )
            rows = []
            for item in parse_record_list(aggregated["_output"].iloc[0]):
                rank = parse_number(item.get("rank"))
                candidate_count = parse_number(
                    item.get("candidate_incident_count")
                )
                successful_count = parse_number(
                    item.get("successful_barrier_count")
                )
                rate = parse_number(item.get("successful_barrier_rate"))
                if (
                    rank is not None
                    and candidate_count is not None
                    and successful_count is not None
                    and rate is not None
                ):
                    rows.append(
                        {
                            "rank": int(rank),
                            "flight_phase": clean_text(item.get("flight_phase")),
                            "aircraft_operator": clean_text(
                                item.get("aircraft_operator")
                            ),
                            "candidate_incident_count": int(candidate_count),
                            "successful_barrier_count": int(successful_count),
                            "successful_barrier_rate": float(rate),
                        }
                    )
            projected = pd.DataFrame.from_records(
                rows,
                columns=[
                    "rank",
                    "flight_phase",
                    "aircraft_operator",
                    "candidate_incident_count",
                    "successful_barrier_count",
                    "successful_barrier_rate",
                ],
            ).sort_values("rank", kind="stable", ignore_index=True)
            step.set_output(projected)

        tracker.record(
            "PROJECT(rows)",
            len(projected),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
