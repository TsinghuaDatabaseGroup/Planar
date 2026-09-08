#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-016."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    normalize_enum,
    parse_record_list,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-016"
COMMUNICATION_PAIRS = ("flight_crew_atc", "flight_crew_flight_crew")
BREAKDOWN_MODES = (
    "phraseology",
    "timing_or_late_information",
    "assumption_or_expectation",
    "task_division_or_role",
    "failure_to_challenge",
)


def pair_is_supported(communication_pair, records) -> bool:
    if communication_pair == "flight_crew_atc":
        return any(
            (
                "Party1 Flight Crew" in record
                and "Party2 ATC" in record
            )
            or (
                "Party1 ATC" in record
                and "Party2 Flight Crew" in record
            )
            for record in records
        )
    if communication_pair == "flight_crew_flight_crew":
        return any(
            "Party1 Flight Crew" in record
            and "Party2 Flight Crew" in record
            for record in records
        )
    return False


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "primary_problem"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        human_factors = incidents[
            incidents["primary_problem"] == "Human Factors"
        ].copy()
        tracker.record(
            "FILTER(primary_problem='Human Factors')",
            len(incidents),
            len(human_factors),
            output=human_factors,
        )

        reports = load_selected_texts("asrs", human_factors)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(human_factors),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(participant groups and primary breakdown modes)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "communication_pairs": (
                        "a JSON array with one object for every clearly supported "
                        "participant group. Each object must contain "
                        "communication_pair as flight_crew_atc or "
                        "flight_crew_flight_crew, and one clearly primary "
                        "breakdown_mode from phraseology, "
                        "timing_or_late_information, assumption_or_expectation, "
                        "task_division_or_role, failure_to_challenge. Use an empty "
                        "array when neither group and mechanism are clearly supported"
                    )
                },
            )
            semantic_rows = []
            for report in raw.to_dict(orient="records"):
                for pair_record in parse_record_list(
                    report.get("communication_pairs")
                ):
                    communication_pair = normalize_enum(
                        pair_record.get("communication_pair"),
                        COMMUNICATION_PAIRS,
                    )
                    breakdown_mode = normalize_enum(
                        pair_record.get("breakdown_mode"),
                        BREAKDOWN_MODES,
                    )
                    if communication_pair is None or breakdown_mode is None:
                        continue
                    semantic_rows.append(
                        {
                            "incident_id": report["incident_id"],
                            "communication_pair": communication_pair,
                            "breakdown_mode": breakdown_mode,
                        }
                    )
            semantic_pairs = pd.DataFrame.from_records(
                semantic_rows,
                columns=[
                    "incident_id",
                    "communication_pair",
                    "breakdown_mode",
                ],
            ).drop_duplicates(
                subset=["incident_id", "communication_pair"],
                keep="first",
            )
            semantic_pairs = semantic_pairs.reset_index(drop=True)
            step.set_output(semantic_pairs)

        person_factors = load_table("asrs", "person_factors.csv")[
            ["incident_id", "factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv)",
            None,
            len(person_factors),
            output=person_factors,
        )

        communication_rows = person_factors[
            (person_factors["factor_type"] == "communication_breakdown")
            & person_factors["value"].str.contains(
                "Flight Crew",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(communication_breakdown AND value CONTAINS Flight Crew)",
            len(person_factors),
            len(communication_rows),
            output=communication_rows,
        )

        communication_pair_records = (
            communication_rows.groupby("incident_id", sort=False)
            .agg(communication_pair_records=("value", list))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(communication_pair_records))",
            len(communication_rows),
            len(communication_pair_records),
            output=communication_pair_records,
        )

        joined_candidates = semantic_pairs.merge(
            communication_pair_records,
            on="incident_id",
            how="inner",
        )
        supported_mask = [
            pair_is_supported(pair, records)
            for pair, records in zip(
                joined_candidates["communication_pair"],
                joined_candidates["communication_pair_records"],
                strict=True,
            )
        ]
        joined = joined_candidates.loc[supported_mask].reset_index(drop=True)
        tracker.record(
            "JOIN(semantic_pairs, communication_pair_records, incident_id and pair)",
            {
                "left": len(semantic_pairs),
                "right": len(communication_pair_records),
            },
            len(joined),
            output=joined,
        )

        grouped = (
            joined.groupby(
                ["communication_pair", "breakdown_mode"],
                sort=False,
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        pair_order = {
            label: index for index, label in enumerate(COMMUNICATION_PAIRS)
        }
        mode_order = {
            label: index for index, label in enumerate(BREAKDOWN_MODES)
        }
        grouped = (
            grouped.assign(
                _pair_order=grouped["communication_pair"].map(pair_order),
                _mode_order=grouped["breakdown_mode"].map(mode_order),
            )
            .sort_values(["_pair_order", "_mode_order"])
            .drop(columns=["_pair_order", "_mode_order"])
            .reset_index(drop=True)
        )
        tracker.record(
            "GROUP_BY([communication_pair, breakdown_mode], COUNT_DISTINCT)",
            len(joined),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
