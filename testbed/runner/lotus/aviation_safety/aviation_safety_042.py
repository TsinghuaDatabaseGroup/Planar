#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-042."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_selected_texts,
    load_table,
    parse_record_list,
    save_output,
    setup,
    stable_mode,
)

TASK_ID = "aviation_safety-042"


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        equipment_rows = events[
            events["label"].str.contains(
                "Aircraft Equipment Problem",
                regex=False,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(label CONTAINS Aircraft Equipment Problem)",
            len(events),
            len(equipment_rows),
            output=equipment_rows,
        )

        equipment_events = equipment_rows[
            ["incident_id"]
        ].drop_duplicates().reset_index(drop=True)
        tracker.record(
            "DEDUP([incident_id])",
            len(equipment_rows),
            len(equipment_events),
            output=equipment_events,
        )

        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        equipment_incidents = equipment_events.merge(
            incidents,
            on="incident_id",
            how="inner",
            validate="one_to_one",
        )
        tracker.record(
            "JOIN(equipment_events, incidents, incident_id)",
            {"left": len(equipment_events), "right": len(incidents)},
            len(equipment_incidents),
            output=equipment_incidents,
        )

        reports = load_selected_texts("asrs", equipment_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=equipment_incidents.text_file)",
            len(equipment_incidents),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(narrative aircraft components and detection cues)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                narrative_components = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "narrative_component_term",
                        "first_detection_cue",
                    ]
                )
            else:
                raw = reports.sem_extract(
                    input_cols=["text"],
                    output_cols={
                        "component_records": (
                            "a JSON array with one object for every distinct aircraft "
                            "component actually involved in the reported equipment "
                            "problem. Each object must contain "
                            "narrative_component_term using the report's wording and "
                            "first_detection_cue as the first sign by which the "
                            "problem was detected, in at most six words. Exclude "
                            "components mentioned only as unaffected context; use an "
                            "empty array when none is supported"
                        )
                    },
                )
                component_records = []
                for report in raw.to_dict(orient="records"):
                    for record in parse_record_list(report.get("component_records")):
                        term = clean_text(
                            record.get("narrative_component_term"),
                            default="",
                        )
                        cue = clean_text(
                            record.get("first_detection_cue"),
                            default="",
                        )
                        if not term or not cue:
                            continue
                        component_records.append(
                            {
                                "incident_id": report["incident_id"],
                                "narrative_component_term": term,
                                "first_detection_cue": cue,
                            }
                        )
                narrative_components = pd.DataFrame.from_records(
                    component_records,
                    columns=[
                        "incident_id",
                        "narrative_component_term",
                        "first_detection_cue",
                    ],
                )
            step.set_output(narrative_components)

        components = load_table("asrs", "components.csv")[["attribute", "value"]].copy()
        tracker.record(
            "SCAN_TABLE(components.csv)",
            None,
            len(components),
            output=components,
        )

        vocabulary_rows = components[
            components["attribute"] == "aircraft_component"
        ].copy()
        tracker.record(
            "FILTER(attribute='aircraft_component')",
            len(components),
            len(vocabulary_rows),
            output=vocabulary_rows,
        )

        component_vocabulary = (
            vocabulary_rows[["value"]]
            .drop_duplicates()
            .rename(columns={"value": "canonical_component"})
            .reset_index(drop=True)
        )
        tracker.record(
            "DEDUP([value])",
            len(vocabulary_rows),
            len(component_vocabulary),
            output=component_vocabulary,
        )

        narrative_bindings = narrative_components.copy()
        narrative_bindings["narrative_component_binding"] = (
            "incident_id="
            + narrative_bindings["incident_id"].astype(str)
            + "; narrative_component_term="
            + narrative_bindings["narrative_component_term"].astype(str)
            + "; first_detection_cue="
            + narrative_bindings["first_detection_cue"].astype(str)
        )
        tracker.record(
            "CODE_MAP(bind narrative component inputs)",
            len(narrative_components),
            len(narrative_bindings),
            output=narrative_bindings,
        )

        vocabulary_bindings = component_vocabulary.copy()
        vocabulary_bindings["canonical_component_binding"] = (
            "canonical_component="
            + vocabulary_bindings["canonical_component"].astype(str)
        )
        tracker.record(
            "CODE_MAP(bind canonical component inputs)",
            len(component_vocabulary),
            len(vocabulary_bindings),
            output=vocabulary_bindings,
        )

        with tracker.step(
            "SEM_JOIN(narrative term to canonical component)",
            input_rows={
                "left": len(narrative_bindings),
                "right": len(vocabulary_bindings),
            },
        ) as step:
            if narrative_bindings.empty or vocabulary_bindings.empty:
                semantic_component_matches = pd.DataFrame(
                    columns=[*narrative_bindings.columns, *vocabulary_bindings.columns]
                )
            else:
                semantic_component_matches = narrative_bindings.sem_join(
                    vocabulary_bindings,
                    "Match narrative component {narrative_component_binding} to "
                    "canonical label {canonical_component_binding} only when they "
                    "denote the same aircraft component or a clear part-to-system "
                    "relationship. Exact spelling is unnecessary; incident identity "
                    "and generic aircraft-topic similarity are not evidence of a "
                    "match."
                )
            step.set_output(semantic_component_matches)

        if semantic_component_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "canonical_component",
                    "narrative_component_term",
                    "incident_count",
                    "most_common_detection_cue",
                ]
            )
        else:
            grouped = (
                semantic_component_matches.groupby(
                    ["canonical_component", "narrative_component_term"],
                    sort=False,
                )
                .agg(
                    incident_count=("incident_id", "nunique"),
                    most_common_detection_cue=("first_detection_cue", stable_mode),
                )
                .reset_index()
            )
        tracker.record(
            "GROUP_BY([canonical_component, narrative_component_term], count/mode)",
            len(semantic_component_matches),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["incident_count", "canonical_component", "narrative_component_term"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([incident_count DESC, component/term ASC])",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(20).copy()
        tracker.record("LIMIT(20)", len(ordered), len(limited), output=limited)

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and component mapping fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
