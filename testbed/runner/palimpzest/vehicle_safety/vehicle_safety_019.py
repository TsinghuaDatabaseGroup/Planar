#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-019."""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_text_documents,
    memory_dataset,
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-019"
DATASET = "nhtsa_vehicle_safety"
PRIMARY_TOPICS = (
    "steering",
    "braking",
    "airbag",
    "electrical",
    "powertrain",
    "other",
)
RECALL_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(\d{2})\s*[Vv]\s*-?\s*(\d{3})(?:\s*-?\s*(\d{3}))?(?!\d)"
)


def normalize_recall_numbers(value) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    normalized = []
    seen = set()
    for raw_value in values:
        if raw_value is None:
            continue
        for match in RECALL_NUMBER_PATTERN.finditer(str(raw_value)):
            recall_number = (
                f"{match.group(1)}V{match.group(2)}{match.group(3) or '000'}"
            )
            if recall_number not in seen:
                seen.add(recall_number)
                normalized.append(recall_number)
    return normalized


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        tracker.record("scan", None, reports)

        closed_plan = memory_dataset(TASK_ID, reports).sem_filter(
            filter=(
                "The investigation report indicates that the investigation is "
                "closed, concluded, ended, or has a similar closed-status indicator."
            ),
            depends_on=["body"],
        )
        started = time.time()
        closed_result = closed_plan.run(config)
        closed = result_frame(closed_result, reports)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            closed,
            closed_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-recalls", closed
        ).sem_map(
            cols=[
                {
                    "name": "cited_recall_numbers",
                    "type": list[str],
                    "desc": (
                        "Every NHTSA recall number explicitly cited in the report, "
                        "as a list. Accept forms such as 12V-491, 23V085, and "
                        "23V085000; return an empty list when none is cited."
                    ),
                }
            ],
            desc="Extract all explicitly cited NHTSA recall numbers.",
            depends_on=["body"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            closed,
            ["cited_recall_numbers"],
        )
        extracted["cited_recall_numbers"] = extracted[
            "cited_recall_numbers"
        ].map(normalize_recall_numbers)
        tracker.record_semantic(
            "sem_map",
            len(closed),
            extracted,
            extraction_result,
            time.time() - started,
        )

        with_numbers = extracted.loc[
            extracted["cited_recall_numbers"].map(len).ge(1)
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), with_numbers)

        classification_plan = memory_dataset(
            f"{TASK_ID}-topic", with_numbers
        ).sem_map(
            cols=[
                {
                    "name": "primary_topic",
                    "type": str,
                    "desc": (
                        "Exactly one of steering, braking, airbag, electrical, "
                        "powertrain, or other."
                    ),
                }
            ],
            desc="Classify the report's primary defect topic.",
            depends_on=["body"],
        )
        started = time.time()
        classification_result = classification_plan.run(config)
        classified = result_frame(
            classification_result,
            with_numbers,
            ["primary_topic"],
        )
        classified["primary_topic"] = classified["primary_topic"].map(
            lambda value: normalize_enum(value, PRIMARY_TOPICS) or "other"
        )
        tracker.record_semantic(
            "sem_map",
            len(with_numbers),
            classified,
            classification_result,
            time.time() - started,
        )

        result = classified[
            ["file_id", "primary_topic", "cited_recall_numbers"]
        ].copy()
        tracker.record("project", len(classified), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
