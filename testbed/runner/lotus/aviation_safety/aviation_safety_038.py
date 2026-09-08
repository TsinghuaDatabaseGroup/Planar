#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-038."""

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

TASK_ID = "aviation_safety-038"
FACTOR_LABELS = {
    "Communication Breakdown": "communication_breakdown",
    "Workload": "workload",
    "Confusion": "confusion",
    "Situational Awareness": "situational_awareness",
}


def atomized_factors(frame: pd.DataFrame, include_incident: bool) -> pd.DataFrame:
    records = []
    for row in frame.to_dict(orient="records"):
        raw_value = str(row["value"])
        for source_label, canonical_label in FACTOR_LABELS.items():
            if source_label not in raw_value:
                continue
            record = {
                "structured_factor_value": raw_value,
                "canonical_human_factor": canonical_label,
            }
            if include_incident:
                record["incident_id"] = row["incident_id"]
            records.append(record)
    columns = [
        *(["incident_id"] if include_incident else []),
        "structured_factor_value",
        "canonical_human_factor",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


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

        human_factor_incidents = incidents[
            incidents["primary_problem"] == "Human Factors"
        ].copy()
        tracker.record(
            "FILTER(primary_problem='Human Factors')",
            len(incidents),
            len(human_factor_incidents),
            output=human_factor_incidents,
        )

        reports = load_selected_texts("asrs", human_factor_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(human_factor_incidents),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_EXTRACT(causal human-factor phrases)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                extracted_factors = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "narrative_factor_phrase",
                        "actor",
                        "operational_effect",
                    ]
                )
            else:
                raw = reports.sem_extract(
                    input_cols=["text"],
                    output_cols={
                        "causal_factor_records": (
                            "a JSON array with one object for every distinct human "
                            "factor that the narrative explicitly portrays as "
                            "contributing to the event. Each object must contain "
                            "narrative_factor_phrase, actor, and operational_effect, "
                            "each as a nonempty phrase of at most six words. Exclude "
                            "background traits, consequences presented as causes, "
                            "unsupported structured labels, and factors explicitly "
                            "ruled out. Use an empty array when none qualifies"
                        )
                    },
                )
                extracted_records = []
                for report in raw.to_dict(orient="records"):
                    for record in parse_record_list(
                        report.get("causal_factor_records")
                    ):
                        phrase = clean_text(
                            record.get("narrative_factor_phrase"),
                            default="",
                        )
                        actor = clean_text(record.get("actor"), default="")
                        effect = clean_text(
                            record.get("operational_effect"),
                            default="",
                        )
                        if not phrase or not actor or not effect:
                            continue
                        extracted_records.append(
                            {
                                "incident_id": report["incident_id"],
                                "narrative_factor_phrase": phrase,
                                "actor": actor,
                                "operational_effect": effect,
                            }
                        )
                extracted_factors = pd.DataFrame.from_records(
                    extracted_records,
                    columns=[
                        "incident_id",
                        "narrative_factor_phrase",
                        "actor",
                        "operational_effect",
                    ],
                )
            step.set_output(extracted_factors)

        person_factors = load_table("asrs", "person_factors.csv")[
            ["incident_id", "factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv AS incident records)",
            None,
            len(person_factors),
            output=person_factors,
        )

        target_factor_rows = person_factors[
            (person_factors["factor_type"] == "human_factors")
            & person_factors["value"].str.contains(
                "|".join(FACTOR_LABELS),
                regex=True,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(human_factors AND value CONTAINS target factor)",
            len(person_factors),
            len(target_factor_rows),
            output=target_factor_rows,
        )

        incident_factor_records = atomized_factors(
            target_factor_rows,
            include_incident=True,
        )
        incident_factor_records = incident_factor_records.rename(
            columns={"canonical_human_factor": "recorded_canonical_factor"}
        )
        tracker.record(
            "PROJECT(atomize incident factor records)",
            len(target_factor_rows),
            len(incident_factor_records),
            output=incident_factor_records,
        )

        factor_candidates = extracted_factors.merge(
            incident_factor_records,
            on="incident_id",
            how="inner",
        )
        tracker.record(
            "JOIN(extracted_factors, incident_factor_records, incident_id)",
            {
                "left": len(extracted_factors),
                "right": len(incident_factor_records),
            },
            len(factor_candidates),
            output=factor_candidates,
        )

        vocabulary_source = load_table("asrs", "person_factors.csv")[
            ["factor_type", "value"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(person_factors.csv AS factor vocabulary)",
            None,
            len(vocabulary_source),
            output=vocabulary_source,
        )

        vocabulary_rows = vocabulary_source[
            (vocabulary_source["factor_type"] == "human_factors")
            & vocabulary_source["value"].str.contains(
                "|".join(FACTOR_LABELS),
                regex=True,
                na=False,
            )
        ].copy()
        tracker.record(
            "FILTER(vocabulary human_factors AND value CONTAINS target factor)",
            len(vocabulary_source),
            len(vocabulary_rows),
            output=vocabulary_rows,
        )

        atomized_vocabulary = atomized_factors(
            vocabulary_rows,
            include_incident=False,
        )
        tracker.record(
            "PROJECT(atomize global factor vocabulary)",
            len(vocabulary_rows),
            len(atomized_vocabulary),
            output=atomized_vocabulary,
        )

        factor_vocabulary = (
            atomized_vocabulary[
                ["canonical_human_factor"]
            ].drop_duplicates().reset_index(drop=True)
        )
        factor_vocabulary["factor_vocabulary"] = factor_vocabulary[
            "canonical_human_factor"
        ].map(
            {canonical: source for source, canonical in FACTOR_LABELS.items()}
        )
        tracker.record(
            "DEDUP([canonical_human_factor])",
            len(atomized_vocabulary),
            len(factor_vocabulary),
            output=factor_vocabulary,
        )

        candidate_bindings = factor_candidates.copy()
        candidate_bindings["narrative_factor_binding"] = (
            "incident_id="
            + candidate_bindings["incident_id"].astype(str)
            + "; narrative_factor_phrase="
            + candidate_bindings["narrative_factor_phrase"].astype(str)
            + "; actor="
            + candidate_bindings["actor"].astype(str)
            + "; operational_effect="
            + candidate_bindings["operational_effect"].astype(str)
            + "; incident_structured_factor="
            + candidate_bindings["structured_factor_value"].astype(str)
            + "; recorded_canonical_factor="
            + candidate_bindings["recorded_canonical_factor"].astype(str)
        )
        tracker.record(
            "CODE_MAP(bind narrative factor candidates)",
            len(factor_candidates),
            len(candidate_bindings),
            output=candidate_bindings,
        )

        vocabulary_bindings = factor_vocabulary.copy()
        vocabulary_bindings["factor_vocabulary_binding"] = (
            "factor_vocabulary="
            + vocabulary_bindings["factor_vocabulary"].astype(str)
            + "; canonical_human_factor="
            + vocabulary_bindings["canonical_human_factor"].astype(str)
        )
        tracker.record(
            "CODE_MAP(bind canonical factor vocabulary)",
            len(factor_vocabulary),
            len(vocabulary_bindings),
            output=vocabulary_bindings,
        )

        with tracker.step(
            "SEM_JOIN(narrative phrase to canonical human factor)",
            input_rows={
                "left": len(candidate_bindings),
                "right": len(vocabulary_bindings),
            },
        ) as step:
            if candidate_bindings.empty or vocabulary_bindings.empty:
                semantic_factor_matches = pd.DataFrame(
                    columns=[*candidate_bindings.columns, *vocabulary_bindings.columns]
                )
            else:
                semantic_factor_matches = candidate_bindings.sem_join(
                    vocabulary_bindings,
                    "Match narrative factor {narrative_factor_binding} to global "
                    "factor {factor_vocabulary_binding} only when the narrative "
                    "phrase semantically expresses that one canonical factor as a "
                    "cause or contributor. The right canonical factor must equal "
                    "the left recorded_canonical_factor and be compatible with its "
                    "incident structured value. Communication breakdown means "
                    "failed or incorrect information exchange; workload means "
                    "excessive or competing task demand; confusion means uncertainty "
                    "or mistaken interpretation; situational awareness means failure "
                    "to perceive, comprehend, or anticipate the situation. Exact "
                    "wording is unnecessary, but merely related or ambiguous "
                    "concepts do not match."
                )
            step.set_output(semantic_factor_matches)

        if semantic_factor_matches.empty:
            grouped = pd.DataFrame(
                columns=[
                    "canonical_human_factor",
                    "incident_count",
                    "most_common_actor",
                    "most_common_operational_effect",
                ]
            )
        else:
            grouped = (
                semantic_factor_matches.groupby(
                    "canonical_human_factor",
                    sort=False,
                )
                .agg(
                    incident_count=("incident_id", "nunique"),
                    most_common_actor=("actor", stable_mode),
                    most_common_operational_effect=(
                        "operational_effect",
                        stable_mode,
                    ),
                )
                .reset_index()
                .sort_values("canonical_human_factor")
                .reset_index(drop=True)
            )
        tracker.record(
            "GROUP_BY([canonical_human_factor], count and modes)",
            len(semantic_factor_matches),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
