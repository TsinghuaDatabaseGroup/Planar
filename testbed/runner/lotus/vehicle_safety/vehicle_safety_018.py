#!/usr/bin/env python3
"""
vehicle_safety-018
Find investigation reports that mention a Tesla model and extract every cited
NHTSA recall number as a deduplicated normalized long-form list.
DAG: SCAN_DOCS(investigation_reports/*.txt) -> SEM_FILTER(Tesla model) ->
     SEM_EXTRACT(cited_recall_numbers) -> FILTER(nonempty) -> PROJECT
Output: table, metric: table_f1
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import StepTracker, Timer, df_records, load_docs, save_output, setup

TASK_ID = "vehicle_safety-018"


def normalize_recall_numbers(value) -> list[str]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    normalized = []
    seen = set()
    for raw_value in values:
        compact = re.sub(r"[^0-9Vv]", "", str(raw_value)).upper()
        match = re.fullmatch(r"(\d{2})V(\d{3})(\d{3})?", compact)
        if match is None:
            continue
        recall_number = f"{match.group(1)}V{match.group(2)}{match.group(3) or '000'}"
        if recall_number not in seen:
            seen.add(recall_number)
            normalized.append(recall_number)
    return normalized


def main():
    setup(max_tokens=512)
    tracker = StepTracker()

    with Timer() as timer:
        reports = load_docs(
            "nhtsa_vehicle_safety", "investigation_reports"
        ).rename(columns={"doc_id": "file_id", "contents": "body"})
        reports = reports[reports["file_id"].str.lower().str.endswith(".txt")]
        tracker.record(
            "SCAN_DOCS(investigation_reports, selector='*.txt')", None, len(reports)
        )

        with tracker.step(
            "SEM_FILTER(explicitly mentions a Tesla model)", input_rows=len(reports)
        ) as step:
            tesla_reports = reports.sem_filter(
                "The investigation report {body} explicitly mentions a Tesla model, "
                "such as Model 3, Model S, Model Y, or Model X."
            )
            step.set_output(tesla_reports)

        with tracker.step(
            "SEM_EXTRACT(cited_recall_numbers)", input_rows=len(tesla_reports)
        ) as step:
            extracted = tesla_reports.sem_extract(
                input_cols=["body"],
                output_cols={
                    "cited_recall_numbers": (
                        "A JSON list containing every NHTSA recall number explicitly "
                        "cited in the report. Accept forms such as 12V-491, 23V085, "
                        "and 23V085000. Normalize each to the long YYVNNN000 form "
                        "and remove duplicates. Return an empty list when none is cited."
                    )
                },
            )
            extracted["cited_recall_numbers"] = extracted[
                "cited_recall_numbers"
            ].apply(normalize_recall_numbers)
            step.set_output(extracted)

        with_numbers = extracted[
            extracted["cited_recall_numbers"].map(len) >= 1
        ]
        tracker.record(
            "FILTER(len(cited_recall_numbers)>=1)",
            len(extracted),
            len(with_numbers),
        )

        result = with_numbers[["file_id", "cited_recall_numbers"]]
        tracker.record(
            "PROJECT([file_id, cited_recall_numbers])",
            len(with_numbers),
            len(result),
        )
        answer = df_records(result)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
