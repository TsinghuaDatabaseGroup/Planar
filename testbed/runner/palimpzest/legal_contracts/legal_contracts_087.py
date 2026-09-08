#!/usr/bin/env python3
"""Palimpzest pipeline for legal_contracts-087."""

from __future__ import annotations

import os
import re
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
    normalize_text_value,
    parse_bool,
    result_frame,
    save_output,
)

TASK_ID = "legal_contracts-087"
DATASET = "contract-exhibit"
QUALIFICATION_FLAGS = [
    "return_or_destroy",
    "injunctive_relief",
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "consent",
]


def normalize_text(value) -> str | None:
    return normalize_text_value(value)


def normalize_title(value) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()) or None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        documents = load_mixed_documents(DATASET)
        tracker.record("scan", None, documents)

        scope_plan = memory_dataset(TASK_ID, documents).sem_filter(
            filter=(
                "Keep the document only if its exhibit type is exactly NDA, not a "
                "dual-labelled exhibit, and it has an identifiable agreement title."
            ),
            depends_on=["text"],
        )
        started = time.time()
        scope_result = scope_plan.run(config)
        scoped = result_frame(scope_result, documents)
        tracker.record_semantic(
            "sem_filter", len(documents), scoped, scope_result, time.time() - started
        )

        definition_plan = memory_dataset(
            f"{TASK_ID}-definition", scoped
        ).sem_filter(
            filter=(
                "Keep the agreement only if it explicitly defines confidential "
                "information."
            ),
            depends_on=["text"],
        )
        started = time.time()
        definition_result = definition_plan.run(config)
        defined = result_frame(definition_result, scoped)
        tracker.record_semantic(
            "sem_filter", len(scoped), defined, definition_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-extraction", defined
        ).sem_map(
            cols=[
                {"name": "agreement_title", "type": str,
                 "desc": "The identifiable agreement title stated in the document."},
                {"name": "return_or_destroy", "type": bool,
                 "desc": "True if return or destruction of confidential materials is required; otherwise false."},
                {"name": "injunctive_relief", "type": bool,
                 "desc": "True if injunctive or equitable relief is provided; otherwise false."},
                {"name": "public_information", "type": bool,
                 "desc": "True if an operative public-information exception is present; otherwise false."},
                {"name": "prior_knowledge", "type": bool,
                 "desc": "True if an operative prior-knowledge exception is present; otherwise false."},
                {"name": "independent_development", "type": bool,
                 "desc": "True if an operative independent-development exception is present; otherwise false."},
                {"name": "third_party_receipt", "type": bool,
                 "desc": "True if an operative lawful third-party-receipt exception is present; otherwise false."},
                {"name": "legal_compulsion", "type": bool,
                 "desc": "True if an operative legal-compulsion exception is present; otherwise false."},
                {"name": "consent", "type": bool,
                 "desc": "True if an operative disclosure-with-consent exception is present; otherwise false."},
            ],
            desc=(
                "Extract the agreement title and determine the return-or-destroy, "
                "injunctive-relief, and six confidentiality-exception flags."
            ),
            depends_on=["text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            defined,
            ["agreement_title", *QUALIFICATION_FLAGS],
        )
        extracted["agreement_title"] = extracted["agreement_title"].map(normalize_text)
        for column in QUALIFICATION_FLAGS:
            extracted[column] = extracted[column].map(parse_bool)
        tracker.record_semantic(
            "sem_map", len(defined), extracted, extraction_result, time.time() - started
        )

        projected = pd.DataFrame(
            {
                "normalized_agreement_title": extracted["agreement_title"].map(
                    normalize_title
                ),
                "qualifies": extracted[QUALIFICATION_FLAGS].all(axis=1).astype(int),
            }
        ).reset_index(drop=True)
        tracker.record("project", len(extracted), projected)

        grouped = (
            projected.groupby("normalized_agreement_title", sort=False)
            .agg(
                family_size=("normalized_agreement_title", "size"),
                qualifying_member_count=("qualifies", "sum"),
            )
            .reset_index()
        )
        tracker.record("groupby", len(projected), grouped)

        qualifying = grouped.loc[
            grouped["family_size"].ge(5)
            & grouped["qualifying_member_count"].ge(1)
        ].reset_index(drop=True)
        tracker.record("filter", len(grouped), qualifying)

        ordered = qualifying.sort_values(
            ["qualifying_member_count", "family_size", "normalized_agreement_title"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record("orderby", len(qualifying), ordered)
        answer = df_records(ordered)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
