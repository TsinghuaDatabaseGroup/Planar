#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-035."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_selected_texts,
    load_table,
    parse_label_list,
    save_output,
    setup,
)

TASK_ID = "aviation_safety-035"
LIGHT_GROUPS = ("low_light", "daylight")
VISUAL_FACTORS = (
    "reduced_visibility",
    "inadequate_or_confusing_lighting",
    "glare",
    "missed_visual_scan",
    "obscured_signage_or_markings",
)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            [
                "incident_id",
                "text_file",
                "aircraft_operator",
                "locale_reference_type",
            ]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        incident_candidates = incidents[
            (incidents["aircraft_operator"] == "Air Carrier")
            & (incidents["locale_reference_type"] == "Airport")
        ].copy()
        tracker.record(
            "FILTER(aircraft_operator='Air Carrier' AND locale='Airport')",
            len(incidents),
            len(incident_candidates),
            output=incident_candidates,
        )

        reports = load_selected_texts("asrs", incident_candidates)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(incident_candidates),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(materially contributing visual factors)",
            input_rows=len(reports),
        ) as step:
            raw = reports.sem_extract(
                input_cols=["text"],
                output_cols={
                    "visual_factors": (
                        "a JSON array containing every visual factor that materially "
                        "contributed to the ground conflict, chosen only from "
                        "reduced_visibility, inadequate_or_confusing_lighting, glare, "
                        "missed_visual_scan, obscured_signage_or_markings. Do not "
                        "include incidental mentions; use an empty array if none"
                    )
                },
            )
            raw["visual_factors"] = raw["visual_factors"].map(
                lambda value: parse_label_list(value, VISUAL_FACTORS)
            )
            supported = raw.loc[
                raw["visual_factors"].map(bool),
                ["incident_id", "visual_factors"],
            ]
            semantic_factors = (
                supported.explode("visual_factors")
                .rename(columns={"visual_factors": "visual_factor"})
                .reset_index(drop=True)
            )
            step.set_output(semantic_factors)

        environment = load_table("asrs", "environment_attributes.csv")[
            ["incident_id", "attribute", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(environment_attributes.csv)",
            None,
            len(environment),
            output=environment,
        )

        light_rows = environment[
            (environment["attribute"] == "light")
            & environment["value"].isin(["Night", "Dusk", "Dawn", "Daylight"])
        ].copy()
        tracker.record(
            "FILTER(attribute='light' AND value IN target light values)",
            len(environment),
            len(light_rows),
            output=light_rows,
        )

        light_records = light_rows[["incident_id"]].copy()
        light_records["light_group"] = light_rows["value"].map(
            lambda value: (
                "low_light" if value in {"Night", "Dusk", "Dawn"} else "daylight"
            )
        )
        tracker.record(
            "PROJECT(light_group=CASE light value)",
            len(light_rows),
            len(light_records),
            output=light_records,
        )

        light_records = light_records.drop_duplicates(
            subset=["incident_id", "light_group"]
        ).reset_index(drop=True)
        tracker.record(
            "DEDUP([incident_id, light_group])",
            len(light_rows),
            len(light_records),
            output=light_records,
        )

        factors_with_light = semantic_factors.merge(
            light_records,
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "JOIN(semantic_factors, light_records, incident_id)",
            {
                "left": len(semantic_factors),
                "right": len(light_records),
            },
            len(factors_with_light),
            output=factors_with_light,
        )

        events = load_table("asrs", "events.csv")[["incident_id", "label"]].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        ground_rows = events[
            (events["label"] == "Conflict Ground Conflict")
            | events["label"].str.startswith("Ground Incursion ", na=False)
            | events["label"].str.startswith("Ground Excursion ", na=False)
        ].copy()
        tracker.record(
            "FILTER(ground conflict/incursion/excursion labels)",
            len(events),
            len(ground_rows),
            output=ground_rows,
        )

        ground_events = ground_rows[["incident_id"]].drop_duplicates().reset_index(
            drop=True
        )
        tracker.record(
            "DEDUP([incident_id])",
            len(ground_rows),
            len(ground_events),
            output=ground_events,
        )

        joined = factors_with_light.merge(
            ground_events,
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "JOIN(factors_with_light, ground_events, incident_id)",
            {
                "left": len(factors_with_light),
                "right": len(ground_events),
            },
            len(joined),
            output=joined,
        )

        grouped = (
            joined.groupby(["light_group", "visual_factor"], sort=False)
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        light_order = {
            label: index for index, label in enumerate(LIGHT_GROUPS)
        }
        factor_order = {
            label: index for index, label in enumerate(VISUAL_FACTORS)
        }
        grouped = (
            grouped.assign(
                _light_order=grouped["light_group"].map(light_order),
                _factor_order=grouped["visual_factor"].map(factor_order),
            )
            .sort_values(["_light_order", "_factor_order"])
            .drop(columns=["_light_order", "_factor_order"])
            .reset_index(drop=True)
        )
        tracker.record(
            "GROUP_BY([light_group, visual_factor], COUNT_DISTINCT)",
            len(joined),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
