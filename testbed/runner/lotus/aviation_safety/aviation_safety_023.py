#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-023."""

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
    save_output,
    setup,
)


TASK_ID = "aviation_safety-023"
FACTOR_ORDER = (
    "communication_breakdown",
    "workload",
    "confusion",
    "situational_awareness",
)


def main():
    setup(max_tokens=384)
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
        human_factors_incidents = incidents[
            incidents["primary_problem"] == "Human Factors"
        ].copy()
        tracker.record(
            "FILTER(primary_problem='Human Factors')",
            len(incidents),
            len(human_factors_incidents),
            output=human_factors_incidents,
        )
        incident_docs = load_selected_texts("asrs", human_factors_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(human_factors_incidents),
            len(incident_docs),
            output=incident_docs,
        )

        communication_source = load_table("asrs", "person_factors.csv")[
            ["incident_id", "factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv) AS communication_rows",
            None,
            len(communication_source),
            output=communication_source,
        )
        communication_rows = communication_source[
            (communication_source["factor_type"] == "human_factors")
            & communication_source["value"].str.contains(
                "Communication Breakdown", na=False, regex=False
            )
        ][["incident_id"]].copy()
        communication_rows["canonical_human_factor"] = "communication_breakdown"
        tracker.record(
            "FILTER/PROJECT(Communication Breakdown)",
            len(communication_source),
            len(communication_rows),
            output=communication_rows,
        )

        workload_source = load_table("asrs", "person_factors.csv")[
            ["incident_id", "factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv) AS workload_rows",
            None,
            len(workload_source),
            output=workload_source,
        )
        workload_rows = workload_source[
            (workload_source["factor_type"] == "human_factors")
            & workload_source["value"].str.contains(
                "Workload", na=False, regex=False
            )
        ][["incident_id"]].copy()
        workload_rows["canonical_human_factor"] = "workload"
        tracker.record(
            "FILTER/PROJECT(Workload)",
            len(workload_source),
            len(workload_rows),
            output=workload_rows,
        )

        confusion_source = load_table("asrs", "person_factors.csv")[
            ["incident_id", "factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv) AS confusion_rows",
            None,
            len(confusion_source),
            output=confusion_source,
        )
        confusion_rows = confusion_source[
            (confusion_source["factor_type"] == "human_factors")
            & confusion_source["value"].str.contains(
                "Confusion", na=False, regex=False
            )
        ][["incident_id"]].copy()
        confusion_rows["canonical_human_factor"] = "confusion"
        tracker.record(
            "FILTER/PROJECT(Confusion)",
            len(confusion_source),
            len(confusion_rows),
            output=confusion_rows,
        )

        awareness_source = load_table("asrs", "person_factors.csv")[
            ["incident_id", "factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv) AS situational_awareness_rows",
            None,
            len(awareness_source),
            output=awareness_source,
        )
        awareness_rows = awareness_source[
            (awareness_source["factor_type"] == "human_factors")
            & awareness_source["value"].str.contains(
                "Situational Awareness", na=False, regex=False
            )
        ][["incident_id"]].copy()
        awareness_rows["canonical_human_factor"] = "situational_awareness"
        tracker.record(
            "FILTER/PROJECT(Situational Awareness)",
            len(awareness_source),
            len(awareness_rows),
            output=awareness_rows,
        )

        factor_rows = pd.concat(
            [
                communication_rows,
                workload_rows,
                confusion_rows,
                awareness_rows,
            ],
            ignore_index=True,
        )
        tracker.record(
            "UNION(factor branches)",
            {
                "communication": len(communication_rows),
                "workload": len(workload_rows),
                "confusion": len(confusion_rows),
                "situational_awareness": len(awareness_rows),
            },
            len(factor_rows),
            output=factor_rows,
        )
        dedup_input_rows = len(factor_rows)
        factor_rows = factor_rows.drop_duplicates(
            subset=["incident_id", "canonical_human_factor"],
            ignore_index=True,
        )
        tracker.record(
            "DEDUP(incident_id, canonical_human_factor)",
            dedup_input_rows,
            len(factor_rows),
            output=factor_rows,
        )

        incident_docs["incident_record"] = (
            "incident_id: "
            + incident_docs["incident_id"].astype(str)
            + "\nnarrative: "
            + incident_docs["text"].fillna("").astype(str)
        )
        incident_bindings = incident_docs[
            ["incident_id", "incident_record"]
        ].rename(columns={"incident_id": "left_incident_id"})
        tracker.record(
            "PROJECT(incident semantic binding)",
            len(incident_docs),
            len(incident_bindings),
            output=incident_bindings,
        )

        factor_rows["factor_record"] = (
            "incident_id: "
            + factor_rows["incident_id"].astype(str)
            + "\nstructured_factor: "
            + factor_rows["canonical_human_factor"]
        )
        factor_bindings = factor_rows[
            ["incident_id", "canonical_human_factor", "factor_record"]
        ].rename(columns={"incident_id": "right_incident_id"})
        tracker.record(
            "PROJECT(factor semantic binding)",
            len(factor_rows),
            len(factor_bindings),
            output=factor_bindings,
        )

        with tracker.step(
            "SEM_JOIN(causally supported incident-factor pairs)",
            input_rows={
                "left": len(incident_bindings),
                "right": len(factor_bindings),
            },
        ) as step:
            supported_pairs = incident_bindings.sem_join(
                factor_bindings,
                "Evaluate incident {incident_record:left} and factor row "
                "{factor_record:right} only as a match when their incident_id "
                "values are the same. Keep the pair only when the narrative "
                "portrays the named structured factor as causally contributing "
                "to the event, rather than merely mentioning it or providing "
                "background context."
            )
            step.set_output(supported_pairs)

        distinct_pairs = supported_pairs[
            ["left_incident_id", "canonical_human_factor"]
        ].drop_duplicates(ignore_index=True)
        grouped = (
            distinct_pairs.groupby("canonical_human_factor", sort=False)
            .size()
            .rename("incident_factor_pair_count")
            .reset_index()
        )
        factor_order = {label: index for index, label in enumerate(FACTOR_ORDER)}
        grouped = grouped.sort_values(
            "canonical_human_factor",
            key=lambda values: values.map(factor_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY(canonical_human_factor, COUNT_DISTINCT(pair))",
            len(supported_pairs),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
