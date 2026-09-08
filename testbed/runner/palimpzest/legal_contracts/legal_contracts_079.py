#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-079."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_mixed_documents,
    memory_dataset,
    normalize_scalar_value,
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-079"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_iso_date(value) -> str | None:
    value = normalize_scalar_value(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def parse_optional_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        agreement_plan = memory_dataset(
            f"{TASK_ID}-agreements", documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is a mutual non-disclosure "
                "agreement, unilateral non-disclosure agreement, or "
                "confidentiality-and-standstill agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        agreement_result = agreement_plan.run(config)
        agreements = result_frame(agreement_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            agreements,
            agreement_result,
            time.time() - started,
        )

        date_plan = memory_dataset(
            f"{TASK_ID}-date", agreements
        ).sem_map(
            cols=[
                {
                    "name": "effective_date",
                    "type": str | None,
                    "desc": (
                        "The explicitly stated effective date as YYYY-MM-DD, "
                        "or null when no effective date is stated."
                    ),
                }
            ],
            desc="Extract the explicitly stated effective date.",
            depends_on=["text"],
        )
        started = time.time()
        date_result = date_plan.run(config)
        dated = result_frame(date_result, agreements, ["effective_date"])
        dated["effective_date"] = dated["effective_date"].map(normalize_iso_date)
        tracker.record_semantic(
            "sem_map", len(agreements), dated, date_result,
            time.time() - started,
        )

        after_2010 = dated.loc[
            dated["effective_date"].notna()
            & (dated["effective_date"] > "2010-12-31")
        ].reset_index(drop=True)
        tracker.record("filter", len(dated), after_2010)

        electronic_plan = memory_dataset(
            f"{TASK_ID}-electronic", after_2010
        ).sem_filter(
            filter=(
                "Keep the agreement only if its protected-information definition "
                "explicitly includes electronic, digital, computer, software, or "
                "comparable computer-related material."
            ),
            depends_on=["text"],
        )
        started = time.time()
        electronic_result = electronic_plan.run(config)
        electronic = result_frame(electronic_result, after_2010)
        tracker.record_semantic(
            "sem_filter",
            len(after_2010),
            electronic,
            electronic_result,
            time.time() - started,
        )

        survival_plan = memory_dataset(
            f"{TASK_ID}-survival", electronic
        ).sem_filter(
            filter=(
                "Keep the agreement only if its confidentiality obligations "
                "expressly continue after agreement termination or after protected "
                "material is returned or destroyed."
            ),
            depends_on=["text"],
        )
        started = time.time()
        survival_result = survival_plan.run(config)
        surviving = result_frame(survival_result, electronic)
        tracker.record_semantic(
            "sem_filter",
            len(electronic),
            surviving,
            survival_result,
            time.time() - started,
        )

        destruction_plan = memory_dataset(
            f"{TASK_ID}-destruction", surviving
        ).sem_filter(
            filter=(
                "Keep the agreement only if the recipient may choose, or may be "
                "directed, to destroy protected material instead of returning "
                "that same material."
            ),
            depends_on=["text"],
        )
        started = time.time()
        destruction_result = destruction_plan.run(config)
        destroyable = result_frame(destruction_result, surviving)
        tracker.record_semantic(
            "sem_filter",
            len(surviving),
            destroyable,
            destruction_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", destroyable
        ).sem_map(
            cols=[
                {
                    "name": "governing_law",
                    "type": str | None,
                    "desc": "The expressly stated governing-law jurisdiction.",
                },
                {
                    "name": "allows_single_archival_copy",
                    "type": bool | None,
                    "desc": (
                        "True only if the agreement expressly permits retention "
                        "of one single archival or record copy. Exclude automatic-"
                        "backup exceptions and permissions to retain an unspecified "
                        "number of copies."
                    ),
                },
            ],
            desc=(
                "Extract governing law and the single-archival-copy permission."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        generated = ["governing_law", "allows_single_archival_copy"]
        extracted = result_frame(extraction_result, destroyable, generated)
        extracted["governing_law"] = extracted["governing_law"].map(normalize_text)
        extracted["allows_single_archival_copy"] = extracted[
            "allows_single_archival_copy"
        ].map(parse_optional_bool)
        tracker.record_semantic(
            "sem_map", len(destroyable), extracted, extraction_result,
            time.time() - started,
        )

        projected = extracted[
            ["document_id", "governing_law", "allows_single_archival_copy"]
        ].reset_index(drop=True)
        tracker.record("project", len(extracted), projected)
        answer = df_records(projected)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
