#!/usr/bin/env python3
"""Palimpzest pipeline for finance-040."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_selected_texts,
    load_table,
    memory_dataset,
    result_frame,
    save_output,
)

TASK_ID = "finance-040"
DATASET = "SEC"
AVAILABLE_YEARS = tuple(range(2019, 2025))
JOIN_COLUMNS = [
    "event_company_name",
    "counterparty_names",
    "registry_company_name",
]


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def _clean_text(value) -> str:
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _string_items(value) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str) and value.strip():
        values = [value]
    else:
        return []

    items = []
    seen = set()
    for value_item in values:
        item = str(value_item).strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            items.append(item)
    return items


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = pd.concat(
            [
                load_table(DATASET, f"CSV/{year}.csv", ["text"])
                for year in AVAILABLE_YEARS
            ],
            ignore_index=True,
        )
        tracker.record("scan", None, filings)

        documents = load_selected_texts(
            DATASET,
            filings,
            path_column="text",
            output_column="document_text",
        )[["document_text"]]
        tracker.record("scan", len(filings), _intermediate_view(documents))

        software_plan = memory_dataset(
            f"{TASK_ID}-software", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if the reporting company's primary industry "
                "sector is Software. Do not qualify a company merely because it "
                "uses or develops some software within a different primary "
                "business."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        software_result = software_plan.run(config)
        software_filings = result_frame(software_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(software_filings),
            software_result,
            time.time() - started,
        )

        event_plan = memory_dataset(
            f"{TASK_ID}-events", software_filings
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The reporting company's legal registrant name.",
                },
                {
                    "name": "counterparty_names",
                    "type": list[str],
                    "desc": (
                        "Distinct legal or commonly stated names of whole-company "
                        "merger or acquisition counterparties disclosed in the "
                        "filing. Exclude asset purchases, minority-stake "
                        "investments, spin-offs, single-company going-private "
                        "recapitalizations, and SPAC or blank-check combinations; "
                        "use an empty list when none qualify."
                    ),
                },
            ],
            desc=(
                "Extract the reporting company name and its qualifying whole-"
                "company merger or acquisition counterparties."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        event_result = event_plan.run(config)
        software_events = result_frame(
            event_result,
            software_filings,
            ["company_name", "counterparty_names"],
        )
        software_events["company_name"] = software_events["company_name"].map(
            _clean_text
        )
        software_events["counterparty_names"] = software_events[
            "counterparty_names"
        ].map(_string_items)
        tracker.record_semantic(
            "sem_map",
            len(software_filings),
            _intermediate_view(software_events),
            event_result,
            time.time() - started,
        )

        candidate_events = software_events.loc[
            software_events["counterparty_names"].map(len).gt(0)
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(software_events),
            _intermediate_view(candidate_events),
        )

        software_registry = (
            software_events[["company_name"]]
            .assign(
                _company_key=lambda frame: frame["company_name"].str.casefold()
            )
            .drop_duplicates("_company_key", keep="first")
            .drop(columns=["_company_key"])
            .rename(columns={"company_name": "registry_company_name"})
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(software_events), software_registry)

        left_bindings = candidate_events[
            ["company_name", "counterparty_names"]
        ].rename(columns={"company_name": "event_company_name"})
        right_bindings = software_registry[["registry_company_name"]].copy()

        join_plan = memory_dataset(
            f"{TASK_ID}-candidate-events", left_bindings
        ).sem_join(
            memory_dataset(f"{TASK_ID}-software-registry", right_bindings),
            condition=(
                "Match only when at least one whole-company M&A counterparty name "
                "identifies the same legal entity as the registry company. Allow "
                "harmless abbreviations, spelling variants, former names, and "
                "legal-suffix differences, but reject merely similar unrelated "
                "names."
            ),
            depends_on=["counterparty_names", "registry_company_name"],
        )
        started = time.time()
        join_result = join_plan.run(config)
        matched = result_frame(join_result)
        if matched.empty:
            matched = pd.DataFrame(columns=JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(left_bindings), "right": len(right_bindings)},
            matched,
            join_result,
            time.time() - started,
        )

        event_names = (
            matched["event_company_name"].fillna("").astype(str).str.casefold()
        )
        registry_names = (
            matched["registry_company_name"].fillna("").astype(str).str.casefold()
        )
        distinct_matches = matched.loc[
            event_names.ne("")
            & registry_names.ne("")
            & event_names.ne(registry_names)
        ].reset_index(drop=True)
        tracker.record("filter", len(matched), distinct_matches)

        pair_rows = []
        for event_name, registry_name in zip(
            distinct_matches["event_company_name"],
            distinct_matches["registry_company_name"],
            strict=False,
        ):
            if event_name.casefold() <= registry_name.casefold():
                company_a, company_b = event_name, registry_name
            else:
                company_a, company_b = registry_name, event_name
            pair_rows.append({"company_a": company_a, "company_b": company_b})
        projected = pd.DataFrame(pair_rows, columns=["company_a", "company_b"])
        tracker.record("project", len(distinct_matches), projected)

        answer_frame = (
            projected.assign(
                _company_a_key=projected["company_a"].str.casefold(),
                _company_b_key=projected["company_b"].str.casefold(),
            )
            .drop_duplicates(["_company_a_key", "_company_b_key"], keep="first")
            .drop(columns=["_company_a_key", "_company_b_key"])
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(projected), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
