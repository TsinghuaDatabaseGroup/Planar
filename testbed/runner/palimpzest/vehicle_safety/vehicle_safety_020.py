#!/usr/bin/env python3
"""Palimpzest pipeline for vehicle_safety-020."""

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
    result_frame,
    save_output,
)

TASK_ID = "vehicle_safety-020"
DATASET = "nhtsa_vehicle_safety"
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

        relevance_plan = memory_dataset(TASK_ID, reports).sem_filter(
            filter=(
                "The investigation report discusses an airbag inflator defect "
                "or an unintended airbag deployment."
            ),
            depends_on=["body"],
        )
        started = time.time()
        relevance_result = relevance_plan.run(config)
        airbag_reports = result_frame(relevance_result, reports)
        tracker.record_semantic(
            "sem_filter",
            len(reports),
            airbag_reports,
            relevance_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-recalls", airbag_reports
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
            airbag_reports,
            ["cited_recall_numbers"],
        )
        extracted["cited_recall_numbers"] = extracted[
            "cited_recall_numbers"
        ].map(normalize_recall_numbers)
        tracker.record_semantic(
            "sem_map",
            len(airbag_reports),
            extracted,
            extraction_result,
            time.time() - started,
        )

        with_numbers = extracted.loc[
            extracted["cited_recall_numbers"].map(len).ge(1)
        ].reset_index(drop=True)
        tracker.record("filter", len(extracted), with_numbers)

        result = with_numbers[["file_id", "cited_recall_numbers"]].copy()
        tracker.record("project", len(with_numbers), result)
        answer = df_records(result)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
