#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-091."""

from __future__ import annotations

import ast
import json
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

TASK_ID = "legal_contracts-091"
DATASET = "contract-exhibit"


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_integer(value) -> int | None:
    value = normalize_scalar_value(value)
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else int(number)


def normalize_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(str(value))
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
        values = parsed if isinstance(parsed, (list, tuple, set)) else [value]
    return [text for item in values if (text := normalize_text(item)) is not None]


def ex21_binding(parent: str | None, subsidiaries: list[str]) -> str:
    return (
        f"parent company: {parent or ''}\n"
        f"explicitly listed subsidiaries: {', '.join(subsidiaries)}"
    )


def main() -> None:
    config = get_config(max_tokens=16384)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        subsidiary_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, subsidiary_documents)

        subsidiary_filter_plan = memory_dataset(
            f"{TASK_ID}-ex21", subsidiary_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-21 subsidiary-list filing "
                "that explicitly identifies its parent company."
            ),
            depends_on=["text"],
        )
        started = time.time()
        subsidiary_filter_result = subsidiary_filter_plan.run(config)
        subsidiary_filings = result_frame(
            subsidiary_filter_result, subsidiary_documents
        )
        tracker.record_semantic(
            "sem_filter", len(subsidiary_documents), subsidiary_filings,
            subsidiary_filter_result, time.time() - started,
        )

        subsidiary_extraction_plan = memory_dataset(
            f"{TASK_ID}-ex21-extraction", subsidiary_filings
        ).sem_map(
            cols=[
                {"name": "parent_company", "type": str,
                 "desc": "The parent company explicitly identified by the filing."},
                {"name": "subsidiary_entities", "type": list[str],
                 "desc": "Every subsidiary legal entity explicitly named in the filing."},
                {"name": "subsidiary_count", "type": int,
                 "desc": "The number of subsidiary legal entities listed in the filing."},
            ],
            desc="Extract the explicit parent, listed subsidiary entities, and count.",
            depends_on=["text"],
        )
        started = time.time()
        subsidiary_extraction_result = subsidiary_extraction_plan.run(config)
        subsidiary_records = result_frame(
            subsidiary_extraction_result,
            subsidiary_filings,
            ["parent_company", "subsidiary_entities", "subsidiary_count"],
        )
        subsidiary_records["parent_company"] = subsidiary_records[
            "parent_company"
        ].map(normalize_text)
        subsidiary_records["subsidiary_entities"] = subsidiary_records[
            "subsidiary_entities"
        ].map(normalize_string_list)
        subsidiary_records["subsidiary_count"] = subsidiary_records[
            "subsidiary_count"
        ].map(normalize_integer)
        tracker.record_semantic(
            "sem_map", len(subsidiary_filings), subsidiary_records,
            subsidiary_extraction_result, time.time() - started,
        )

        subsidiary_projected = subsidiary_records[
            ["parent_company", "subsidiary_count"]
        ].copy()
        subsidiary_projected["ex21_entity_binding"] = [
            ex21_binding(parent, subsidiaries)
            for parent, subsidiaries in zip(
                subsidiary_records["parent_company"],
                subsidiary_records["subsidiary_entities"],
            )
        ]
        subsidiary_projected = subsidiary_projected.reset_index(drop=True)
        tracker.record("project", len(subsidiary_records), subsidiary_projected)

        agreement_documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, agreement_documents)

        agreement_filter_plan = memory_dataset(
            f"{TASK_ID}-ex10", agreement_documents
        ).sem_filter(
            filter=(
                "Keep the document only if it is an EX-10 employment, offer, or "
                "separation agreement."
            ),
            depends_on=["text"],
        )
        started = time.time()
        agreement_filter_result = agreement_filter_plan.run(config)
        ex10_agreements = result_frame(agreement_filter_result, agreement_documents)
        tracker.record_semantic(
            "sem_filter", len(agreement_documents), ex10_agreements,
            agreement_filter_result, time.time() - started,
        )

        restriction_filter_plan = memory_dataset(
            f"{TASK_ID}-restrictions", ex10_agreements
        ).sem_filter(
            filter=(
                "Keep the agreement only if it contains an operative non-compete "
                "clause and at least one operative employee or customer non-"
                "solicitation clause, and states the duration of both matched "
                "restrictions."
            ),
            depends_on=["text"],
        )
        started = time.time()
        restriction_filter_result = restriction_filter_plan.run(config)
        restrictive_agreements = result_frame(
            restriction_filter_result, ex10_agreements
        )
        tracker.record_semantic(
            "sem_filter", len(ex10_agreements), restrictive_agreements,
            restriction_filter_result, time.time() - started,
        )

        agreement_extraction_plan = memory_dataset(
            f"{TASK_ID}-ex10-extraction", restrictive_agreements
        ).sem_map(
            cols=[
                {"name": "ex10_entities", "type": list[str],
                 "desc": "All employer and company entity names stated in the agreement."}
            ],
            desc="Extract all employer and company entities from the EX-10 agreement.",
            depends_on=["text"],
        )
        started = time.time()
        agreement_extraction_result = agreement_extraction_plan.run(config)
        agreement_records = result_frame(
            agreement_extraction_result,
            restrictive_agreements,
            ["ex10_entities"],
        )
        agreement_records["ex10_entities"] = agreement_records[
            "ex10_entities"
        ].map(normalize_string_list)
        tracker.record_semantic(
            "sem_map", len(restrictive_agreements), agreement_records,
            agreement_extraction_result, time.time() - started,
        )

        agreement_projected = agreement_records[["document_id"]].rename(
            columns={"document_id": "ex10_document_id"}
        )
        agreement_projected["ex10_entity_binding"] = agreement_records[
            "ex10_entities"
        ].map(lambda values: ", ".join(values))
        agreement_projected = agreement_projected.reset_index(drop=True)
        tracker.record("project", len(agreement_records), agreement_projected)

        join_columns = [
            "parent_company",
            "subsidiary_count",
            "ex21_entity_binding",
            "ex10_document_id",
            "ex10_entity_binding",
        ]
        if subsidiary_projected.empty or agreement_projected.empty:
            matched = pd.DataFrame(columns=join_columns)
            tracker.record(
                "sem_join",
                {
                    "left": len(subsidiary_projected),
                    "right": len(agreement_projected),
                },
                matched,
            )
        else:
            join_plan = memory_dataset(
                f"{TASK_ID}-ex21-records", subsidiary_projected
            ).sem_join(
                memory_dataset(f"{TASK_ID}-ex10-records", agreement_projected),
                condition=(
                    "At least one employer or company entity in the EX-10 record is "
                    "either the EX-21 parent company or an entity explicitly named in "
                    "that EX-21 subsidiary list, allowing ordinary capitalization, "
                    "punctuation, and corporate-suffix variation."
                ),
                depends_on=["ex21_entity_binding", "ex10_entity_binding"],
            )
            started = time.time()
            join_result = join_plan.run(config)
            matched = result_frame(join_result)
            if matched.empty:
                matched = pd.DataFrame(columns=join_columns)
            tracker.record_semantic(
                "sem_join",
                {
                    "left": len(subsidiary_projected),
                    "right": len(agreement_projected),
                },
                matched,
                join_result,
                time.time() - started,
            )

        grouped = (
            matched.groupby("parent_company", sort=False)
            .agg(
                subsidiary_count=("subsidiary_count", "max"),
                qualifying_agreement_count=("ex10_document_id", "nunique"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(matched), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
