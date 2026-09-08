#!/usr/bin/env python3
"""LOTUS pipeline for aviation_safety-043."""

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

TASK_ID = "aviation_safety-043"
WEATHER_MECHANISMS = (
    "convective_weather",
    "turbulence",
    "windshear",
    "icing",
    "low_visibility",
    "low_ceiling",
    "strong_or_gusty_wind",
    "precipitation",
)
RECORDED_DECISIONS = (
    "diversion",
    "emergency_declaration",
    "new_clearance",
)
DECISION_RESULT_LABELS = {
    "diversion": "Flight Crew Diverted",
    "emergency_declaration": "General Declared Emergency",
    "new_clearance": "Air Traffic Control Issued New Clearance",
}


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        incidents = load_table("asrs", "incidents.csv")[
            ["incident_id", "text_file", "local_time_of_day"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(incidents.csv)",
            None,
            len(incidents),
            output=incidents,
        )

        filtered_incidents = incidents[
            incidents["local_time_of_day"] == "1201-1800"
        ].copy()
        tracker.record(
            "FILTER(local_time_of_day='1201-1800')",
            len(incidents),
            len(filtered_incidents),
            output=filtered_incidents,
        )

        reports = load_selected_texts("asrs", filtered_incidents)[
            ["incident_id", "text"]
        ]
        tracker.record(
            "SCAN_DOCS(selector=filtered_incidents.text_file)",
            len(filtered_incidents),
            len(reports),
            output=reports,
        )

        with tracker.step(
            "SEM_FILTER(weather explicitly caused operational decision)",
            input_rows=len(reports),
        ) as step:
            if reports.empty:
                causal_weather = reports.copy()
            else:
                causal_weather = reports.sem_filter(
                    "Keep report {text} only when weather or visibility is "
                    "explicitly connected as a cause of an operational decision, "
                    "rather than merely co-occurring with the decision."
                )
            step.set_output(causal_weather)

        with tracker.step(
            "SEM_FILTER(exclude incidental weather and comfort-only turbulence)",
            input_rows=len(causal_weather),
        ) as step:
            if causal_weather.empty:
                qualifying_weather = causal_weather.copy()
            else:
                qualifying_weather = causal_weather.sem_filter(
                    "Keep report {text} only when the causal weather is not an "
                    "incidental mention and is not turbulence whose only effect was "
                    "passenger comfort."
                )
            step.set_output(qualifying_weather)

        with tracker.step(
            "SEM_EXTRACT(all causal weather and decision pairs)",
            input_rows=len(qualifying_weather),
        ) as step:
            if qualifying_weather.empty:
                semantic_pairs = pd.DataFrame(
                    columns=[
                        "incident_id",
                        "weather_mechanism",
                        "recorded_decision",
                    ]
                )
            else:
                raw_pairs = qualifying_weather.sem_extract(
                    input_cols=["text"],
                    output_cols={
                        "causal_pairs": (
                            "a JSON array with one object for every distinct, "
                            "explicitly supported causal weather-decision pair. Each "
                            "object must contain weather_mechanism as exactly one of "
                            "convective_weather, turbulence, windshear, icing, "
                            "low_visibility, low_ceiling, strong_or_gusty_wind, "
                            "precipitation and recorded_decision as exactly one of "
                            "diversion, emergency_declaration, new_clearance. Use an "
                            "empty array if no pair is explicit"
                        )
                    },
                )
                pair_records = []
                for report in raw_pairs.to_dict(orient="records"):
                    for record in parse_record_list(report.get("causal_pairs")):
                        mechanism = normalize_enum(
                            record.get("weather_mechanism"),
                            WEATHER_MECHANISMS,
                        )
                        decision = normalize_enum(
                            record.get("recorded_decision"),
                            RECORDED_DECISIONS,
                        )
                        if mechanism is None or decision is None:
                            continue
                        pair_records.append(
                            {
                                "incident_id": report["incident_id"],
                                "weather_mechanism": mechanism,
                                "recorded_decision": decision,
                            }
                        )
                semantic_pairs = pd.DataFrame.from_records(
                    pair_records,
                    columns=[
                        "incident_id",
                        "weather_mechanism",
                        "recorded_decision",
                    ],
                )
            step.set_output(semantic_pairs)

        events = load_table("asrs", "events.csv")[
            ["incident_id", "event_type", "label"]
        ].copy()
        tracker.record(
            "SCAN_TABLE(events.csv)",
            None,
            len(events),
            output=events,
        )

        result_rows = events[
            (events["event_type"] == "result")
            & events["label"].isin(DECISION_RESULT_LABELS.values())
        ].copy()
        tracker.record(
            "FILTER(event_type='result' AND label IN target results)",
            len(events),
            len(result_rows),
            output=result_rows,
        )

        recorded_results = (
            result_rows.groupby("incident_id", sort=False)
            .agg(
                recorded_result_labels=(
                    "label",
                    lambda values: list(dict.fromkeys(values.dropna())),
                )
            )
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([incident_id], COLLECT(label))",
            len(result_rows),
            len(recorded_results),
            output=recorded_results,
        )

        joined_candidates = semantic_pairs.merge(
            recorded_results,
            on="incident_id",
            how="inner",
            validate="many_to_one",
        )
        matching_result = [
            DECISION_RESULT_LABELS[decision] in labels
            for decision, labels in zip(
                joined_candidates["recorded_decision"],
                joined_candidates["recorded_result_labels"],
                strict=True,
            )
        ]
        joined = joined_candidates.loc[matching_result].reset_index(drop=True)
        tracker.record(
            "JOIN(semantic_pairs, recorded_results, incident_id and decision label)",
            {"left": len(semantic_pairs), "right": len(recorded_results)},
            len(joined),
            output=joined,
        )

        grouped = (
            joined.groupby(
                ["weather_mechanism", "recorded_decision"],
                sort=False,
            )
            .agg(incident_count=("incident_id", "nunique"))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY([weather_mechanism, recorded_decision], COUNT_DISTINCT)",
            len(joined),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["incident_count", "weather_mechanism", "recorded_decision"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY([incident_count DESC, mechanism/decision ASC])",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(20).copy()
        tracker.record("LIMIT(20)", len(ordered), len(limited), output=limited)

        result = limited.copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        tracker.record(
            "PROJECT(rank and weather-decision fields)",
            len(limited),
            len(result),
            output=result,
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
