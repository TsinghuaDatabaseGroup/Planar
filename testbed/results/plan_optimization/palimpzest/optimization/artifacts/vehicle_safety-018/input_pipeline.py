#!/usr/bin/env python3
"""Plan-optimization pipeline for vehicle_safety-018."""

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
    run_plan_optimization,
    save_output,
)

TASK_ID = "vehicle_safety-018"
DATASET = "nhtsa_vehicle_safety"
RECALL_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(\d{2})\s*[Vv]\s*-?\s*(\d{3})(?:\s*-?\s*(\d{3}))?(?!\d)"
)
def has_recall_numbers(record: dict) -> bool:
    value = record.get("normalized_cited_recall_numbers")
    if isinstance(value, (list, tuple, set)):
        return len(value) >= 1
    return bool(str(value or "").strip())


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


def normalize_recall_number_record(record: dict) -> dict:
    return {
        "normalized_cited_recall_numbers": normalize_recall_numbers(
            record.get("cited_recall_numbers")
        )
    }


def main() -> None:
    tracker = StepTracker(TASK_ID, optimizer_strategy="pareto")
    config = get_config(max_tokens=4096)

    with Timer() as timer:
        reports = load_text_documents(
            DATASET,
            "investigation_reports",
            pattern="*.txt",
            id_column="file_id",
            text_column="body",
        )
        plan = memory_dataset(TASK_ID, reports).sem_filter(
            (
                "Keep this investigation report only if it explicitly names a "
                "Tesla Model 3, Model S, Model X, or Model Y. A generic mention "
                "of Tesla without one of those concrete models is insufficient."
            ),
            depends_on=["body"],
        )
        plan = plan.sem_map(
            cols=[
                {
                    "name": "cited_recall_numbers",
                    "type": list[str],
                    "desc": (
                        "Every explicitly cited NHTSA recall number, normalized "
                        "to the long YYVNNNNNN form and deduplicated; return an "
                        "empty list if none is cited."
                    ),
                }
            ],
            desc=(
                "Extract, normalize, and deduplicate all NHTSA recall numbers "
                "cited in the report."
            ),
            depends_on=["body"],
        )
        plan = plan.map(
            normalize_recall_number_record,
            cols=[
                {
                    "name": "normalized_cited_recall_numbers",
                    "type": list[str],
                    "desc": "Validated, normalized, deduplicated recall numbers.",
                }
            ],
            depends_on=["cited_recall_numbers"],
        )
        plan = plan.filter(
            has_recall_numbers,
            depends_on=["normalized_cited_recall_numbers"],
        )
        plan = plan.project(["file_id", "normalized_cited_recall_numbers"])

        started = time.time()
        optimized = run_plan_optimization(
            plan,
            config,
            task_id=TASK_ID,
        )
        if optimized.result is None:
            return
        output = optimized.result.to_df().reset_index(drop=True)
        tracker.record_semantic(
            "optimized_plan",
            len(reports),
            output,
            optimized.result,
            time.time() - started,
        )

        output = output.rename(
            columns={
                "normalized_cited_recall_numbers": "cited_recall_numbers"
            }
        ).reindex(columns=["file_id", "cited_recall_numbers"])
        answer = df_records(output)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
