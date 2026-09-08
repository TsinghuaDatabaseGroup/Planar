#!/usr/bin/env python3
"""
vehicle_safety-040
Group semantically matched Jeep engine complaints and recalls by normalized
failure mode, including timing and complaint-sentiment statistics.
DAG: complaint FILTER -> SEM_JOIN(recall FILTER) -> SEM_EXTRACT -> SEM_FILTER ->
     SEM_CLASSIFY(sentiment) -> GROUP_BY(failure_mode) -> PROJECT
Output: table, metric: f1
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (
    StepTracker,
    Timer,
    df_records,
    load_jsonl,
    load_table,
    save_output,
    setup,
)

TASK_ID = "vehicle_safety-040"


def alpha_ascending_mode(values):
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    max_count = counts.max()
    return sorted(str(value) for value in counts[counts == max_count].index)[0]


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        complaints = load_table("nhtsa_vehicle_safety", "complaints.csv")[
            [
                "complaint_id",
                "make",
                "component_id",
                "received_date",
                "summary_text",
            ]
        ]
        tracker.record("SCAN_TABLE(complaints)", None, len(complaints))

        jeep_engine_complaints = complaints[
            (complaints["make"] == "JEEP")
            & (complaints["component_id"] == "ENGINE")
        ]
        tracker.record(
            "FILTER(make='JEEP' AND component_id='ENGINE')",
            len(complaints),
            len(jeep_engine_complaints),
        )

        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "campaign_report_received_date",
                "vehicle_make",
                "component_component_id",
                "risk_defect_summary",
            ]
        ].rename(
            columns={
                "component_component_id": "component_id",
                "risk_defect_summary": "recall_defect_summary",
            }
        )
        tracker.record("SCAN_DOCS(recalls AS r)", None, len(recalls))

        jeep_engine_recalls = recalls[
            (recalls["vehicle_make"] == "JEEP")
            & (recalls["component_id"] == "ENGINE")
        ]
        tracker.record(
            "FILTER(vehicle.make='JEEP' AND component.component_id='ENGINE')",
            len(recalls),
            len(jeep_engine_recalls),
        )

        complaint_bindings = jeep_engine_complaints.drop(
            columns=["make", "component_id"]
        ).copy()
        complaint_bindings["complaint_recall_binding"] = (
            "component_id=ENGINE; complaint_summary="
            + complaint_bindings["summary_text"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN complaint inputs)",
            len(jeep_engine_complaints),
            len(complaint_bindings),
        )

        recall_bindings = jeep_engine_recalls[
            [
                "campaign_number",
                "campaign_report_received_date",
                "recall_defect_summary",
            ]
        ].copy()
        recall_bindings["recall_binding"] = (
            "component_id=ENGINE; recall_defect="
            + recall_bindings["recall_defect_summary"].fillna("").astype(str)
        )
        tracker.record(
            "CODE_MAP(bind SEM_JOIN recall inputs)",
            len(jeep_engine_recalls),
            len(recall_bindings),
        )

        with tracker.step(
            "SEM_JOIN(Jeep ENGINE complaint and recall share failure mechanism)",
            input_rows={
                "left": len(complaint_bindings),
                "right": len(recall_bindings),
            },
        ) as step:
            if complaint_bindings.empty or recall_bindings.empty:
                matched_pairs = pd.DataFrame(
                    columns=[
                        "complaint_id",
                        "received_date",
                        "summary_text",
                        "campaign_number",
                        "campaign_report_received_date",
                        "recall_defect_summary",
                    ]
                )
            else:
                matched_pairs = complaint_bindings.sem_join(
                    recall_bindings,
                    "Within Jeep ENGINE records, match complaint "
                    "{complaint_recall_binding} to recall {recall_binding} only "
                    "when the complaint narrative and recall defect summary "
                    "describe the same engine failure mechanism."
                )
            step.set_output(matched_pairs)

        with tracker.step(
            "SEM_EXTRACT(normalized failure_mode<=6 words)",
            input_rows=len(matched_pairs),
        ) as step:
            if matched_pairs.empty:
                extracted = matched_pairs.copy()
                extracted["failure_mode"] = pd.Series(dtype="object")
            else:
                extracted = matched_pairs.sem_extract(
                    input_cols=["summary_text", "recall_defect_summary"],
                    output_cols={
                        "failure_mode": (
                            "A normalized engine failure mode in at most 6 words, "
                            "shared by the complaint and recall."
                        )
                    },
                )
                extracted["failure_mode"] = (
                    extracted["failure_mode"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(extracted)

        with tracker.step(
            "SEM_FILTER(engine failure affects driving safety)",
            input_rows=len(extracted),
        ) as step:
            if extracted.empty:
                safety_matches = extracted.copy()
            else:
                safety_matches = extracted.sem_filter(
                    "The matched complaint {summary_text} and recall defect "
                    "{recall_defect_summary} describe an engine failure that can "
                    "affect driving safety, such as stalling, loss of motive power, "
                    "fire risk, or engine seizure."
                )
            step.set_output(safety_matches)

        with tracker.step(
            "SEM_CLASSIFY(complaint_sentiment)",
            input_rows=len(safety_matches),
        ) as step:
            if safety_matches.empty:
                labeled = safety_matches.copy()
                labeled["complaint_sentiment"] = pd.Series(dtype="object")
            else:
                labeled = safety_matches.sem_map(
                    "Assign complaint {summary_text} sentiment. Output exactly one "
                    "label: acute_safety_event, intermittent_issue, or "
                    "cosmetic_or_minor.",
                    suffix="complaint_sentiment",
                )
                labeled["complaint_sentiment"] = (
                    labeled["complaint_sentiment"]
                    .astype(str)
                    .str.strip()
                    .str.strip(".")
                    .str.lower()
                )
            step.set_output(labeled)

        labeled = labeled.copy()
        labeled["_complaint_date"] = pd.to_datetime(
            labeled["received_date"], errors="coerce"
        )
        labeled["_recall_date"] = pd.to_datetime(
            labeled["campaign_report_received_date"], errors="coerce"
        )

        if labeled.empty:
            grouped = pd.DataFrame(
                columns=[
                    "failure_mode",
                    "matched_complaint_count",
                    "distinct_campaign_count",
                    "earliest_complaint_date",
                    "earliest_recall_date",
                    "pre_recall_complaint_count",
                    "dominant_complaint_sentiment",
                ]
            )
        else:
            grouped = labeled.groupby("failure_mode", as_index=False).agg(
                matched_complaint_count=("complaint_id", "nunique"),
                distinct_campaign_count=("campaign_number", "nunique"),
                earliest_complaint_date=("_complaint_date", "min"),
                earliest_recall_date=("_recall_date", "min"),
                dominant_complaint_sentiment=(
                    "complaint_sentiment",
                    alpha_ascending_mode,
                ),
            )
            pre_recall_counts = (
                labeled[labeled["_complaint_date"] < labeled["_recall_date"]]
                .groupby("failure_mode")["complaint_id"]
                .nunique()
                .rename("pre_recall_complaint_count")
            )
            grouped = grouped.merge(
                pre_recall_counts,
                on="failure_mode",
                how="left",
            )
            grouped["pre_recall_complaint_count"] = (
                grouped["pre_recall_complaint_count"].fillna(0).astype(int)
            )
            grouped["earliest_complaint_date"] = grouped[
                "earliest_complaint_date"
            ].dt.strftime("%Y-%m-%d")
            grouped["earliest_recall_date"] = grouped[
                "earliest_recall_date"
            ].dt.strftime("%Y-%m-%d")
        tracker.record(
            "GROUP_BY([failure_mode], matched complaints, campaigns, earliest dates, pre-recall complaints, sentiment mode)",
            len(labeled),
            len(grouped),
        )

        result = grouped[
            [
                "failure_mode",
                "matched_complaint_count",
                "distinct_campaign_count",
                "earliest_complaint_date",
                "earliest_recall_date",
                "pre_recall_complaint_count",
                "dominant_complaint_sentiment",
            ]
        ]
        tracker.record(
            "PROJECT([failure_mode, counts, earliest dates, pre_recall_complaint_count, dominant sentiment])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
