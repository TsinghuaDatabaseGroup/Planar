#!/usr/bin/env python3
"""
vehicle_safety-011
For 2022-2025 recalls, report each manufacturer's dominant corrective-action
style, recall count, and distinct recalled-component count.
DAG: SCAN_DOCS(recalls) -> FILTER(date range) -> DEDUP(campaign) ->
     SEM_CLASSIFY(action style) -> GROUP_BY(manufacturer) -> PROJECT
Output: table, metric: table_f1
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, df_records, load_jsonl, save_output, setup

TASK_ID = "vehicle_safety-011"


def alpha_ascending_mode(values):
    counts = values.value_counts()
    max_count = counts.max()
    return sorted(str(value) for value in counts[counts == max_count].index)[0]


def main():
    setup(max_tokens=128)
    tracker = StepTracker()

    with Timer() as timer:
        recalls = load_jsonl("nhtsa_vehicle_safety", "recalls.jsonl")[
            [
                "campaign_number",
                "campaign_manufacturer",
                "campaign_report_received_date",
                "component_component_id",
                "remedy_corrective_action",
            ]
        ]
        tracker.record("SCAN_DOCS(recalls)", None, len(recalls))

        report_dates = pd.to_datetime(
            recalls["campaign_report_received_date"], errors="coerce"
        )
        in_range = recalls[
            (report_dates >= pd.Timestamp("2022-01-01"))
            & (report_dates <= pd.Timestamp("2025-12-31"))
        ]
        tracker.record(
            "FILTER(report_received_date>='2022-01-01' AND report_received_date<='2025-12-31')",
            len(recalls),
            len(in_range),
        )

        deduped = in_range.drop_duplicates(subset=["campaign_number"])
        tracker.record(
            "DEDUP([campaign.number])", len(in_range), len(deduped)
        )

        with tracker.step(
            "SEM_CLASSIFY(corrective_action_style)", input_rows=len(deduped)
        ) as step:
            classified = deduped.sem_map(
                "Classify the dominant corrective-action style described by "
                "{remedy_corrective_action}. Output exactly one label: "
                "software_update, part_replacement, dealer_inspection_only, or "
                "other. Use dealer_inspection_only only when inspection is the "
                "corrective action without a software update or part replacement.",
                suffix="corrective_action_style",
            )
            classified["corrective_action_style"] = (
                classified["corrective_action_style"]
                .astype(str)
                .str.strip()
                .str.strip(".")
                .str.lower()
            )
            step.set_output(classified)

        grouped = (
            classified.groupby("campaign_manufacturer", as_index=False, dropna=False)
            .agg(
                recall_count=("campaign_number", "nunique"),
                dominant_action_style=(
                    "corrective_action_style",
                    alpha_ascending_mode,
                ),
                distinct_components_recalled=("component_component_id", "nunique"),
            )
        )
        tracker.record(
            "GROUP_BY([campaign.manufacturer], recall_count, mode_with_tiebreak(action_style), distinct_components_recalled)",
            len(classified),
            len(grouped),
        )

        result = grouped.rename(
            columns={"campaign_manufacturer": "manufacturer"}
        )[
            [
                "manufacturer",
                "recall_count",
                "dominant_action_style",
                "distinct_components_recalled",
            ]
        ]
        tracker.record(
            "PROJECT([manufacturer, recall_count, dominant_action_style, distinct_components_recalled])",
            len(grouped),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
