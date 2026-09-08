#!/usr/bin/env python3
"""Palimpzest pipeline for finance-042."""

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

TASK_ID = "finance-042"
DATASET = "SEC"
AVAILABLE_YEARS = tuple(range(2019, 2025))
FINANCE_JOIN_COLUMNS = [
    "company_name",
    "finance_subsidiary_names",
    "filing_texts",
    "finance_company_name",
]
MA_JOIN_COLUMNS = [
    *FINANCE_JOIN_COLUMNS,
    "ma_counterparty_names",
    "ma_company_name",
]


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_text", "filing_texts"],
        errors="ignore",
    )


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
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    return items


def _union_distinct_names(values) -> list[str]:
    names = []
    seen = set()
    for value in values:
        for name in _string_items(value):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


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

        company_plan = memory_dataset(
            f"{TASK_ID}-filings", documents
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The filing company's legal registrant name.",
                },
                {
                    "name": "finance_subsidiary_names",
                    "type": list[str],
                    "desc": (
                        "Distinct legal names of entities that the filing explicitly "
                        "identifies as wholly-owned finance or funding subsidiaries "
                        "of the reporting company; use an empty list when none are "
                        "disclosed."
                    ),
                },
            ],
            desc=(
                "Extract the reporting company legal name and names of its "
                "wholly-owned finance or funding subsidiaries."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        company_result = company_plan.run(config)
        extracted_filings = result_frame(
            company_result,
            documents,
            ["company_name", "finance_subsidiary_names"],
        )
        extracted_filings["company_name"] = extracted_filings[
            "company_name"
        ].map(_clean_text)
        extracted_filings["finance_subsidiary_names"] = extracted_filings[
            "finance_subsidiary_names"
        ].map(_string_items)
        tracker.record_semantic(
            "sem_map",
            len(documents),
            _intermediate_view(extracted_filings),
            company_result,
            time.time() - started,
        )

        company_rows = []
        for company_name, group in extracted_filings.groupby(
            "company_name",
            sort=False,
            dropna=False,
        ):
            company_rows.append(
                {
                    "company_name": company_name,
                    "finance_subsidiary_names": _union_distinct_names(
                        group["finance_subsidiary_names"]
                    ),
                    "filing_texts": group["document_text"].astype(str).tolist(),
                }
            )
        companies = pd.DataFrame(
            company_rows,
            columns=[
                "company_name",
                "finance_subsidiary_names",
                "filing_texts",
            ],
        )
        tracker.record(
            "groupby",
            len(extracted_filings),
            _intermediate_view(companies),
        )

        finance_candidates = companies.loc[
            companies["finance_subsidiary_names"].map(len).gt(0)
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(companies),
            _intermediate_view(finance_candidates),
        )

        finance_registry = companies[["company_name"]].rename(
            columns={"company_name": "finance_company_name"}
        )
        finance_join_plan = memory_dataset(
            f"{TASK_ID}-finance-candidates", finance_candidates
        ).sem_join(
            memory_dataset(f"{TASK_ID}-finance-registry", finance_registry),
            condition=(
                "Match only when at least one wholly-owned finance or funding "
                "subsidiary name identifies the same legal entity as the registry "
                "company. Allow harmless abbreviations, spelling variants, former "
                "names, and legal-suffix differences, but reject merely similar "
                "unrelated names."
            ),
            depends_on=["finance_subsidiary_names", "finance_company_name"],
        )
        started = time.time()
        finance_join_result = finance_join_plan.run(config)
        finance_matches = result_frame(finance_join_result)
        if finance_matches.empty:
            finance_matches = pd.DataFrame(columns=FINANCE_JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(finance_candidates), "right": len(finance_registry)},
            _intermediate_view(finance_matches),
            finance_join_result,
            time.time() - started,
        )

        owner_names = (
            finance_matches["company_name"].fillna("").astype(str).str.casefold()
        )
        finance_names = (
            finance_matches["finance_company_name"]
            .fillna("")
            .astype(str)
            .str.casefold()
        )
        distinct_finance_matches = finance_matches.loc[
            owner_names.ne("")
            & finance_names.ne("")
            & owner_names.ne(finance_names)
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(finance_matches),
            _intermediate_view(distinct_finance_matches),
        )

        ma_plan = memory_dataset(
            f"{TASK_ID}-ma-candidates", distinct_finance_matches
        ).sem_map(
            cols=[
                {
                    "name": "ma_counterparty_names",
                    "type": list[str],
                    "desc": (
                        "Distinct legal or commonly stated names of whole-company "
                        "merger or acquisition counterparties involving the "
                        "reporting company across these filings. Exclude asset "
                        "purchases, minority-stake investments, spin-offs, and SPAC "
                        "or blank-check business combinations; use an empty list "
                        "when none qualify."
                    ),
                }
            ],
            desc=(
                "Extract qualifying whole-company merger or acquisition "
                "counterparty names from the reporting company's filing texts."
            ),
            depends_on=["filing_texts"],
        )
        started = time.time()
        ma_result = ma_plan.run(config)
        ma_candidates = result_frame(
            ma_result,
            distinct_finance_matches,
            ["ma_counterparty_names"],
        )
        ma_candidates["ma_counterparty_names"] = ma_candidates[
            "ma_counterparty_names"
        ].map(_string_items)
        tracker.record_semantic(
            "sem_map",
            len(distinct_finance_matches),
            _intermediate_view(ma_candidates),
            ma_result,
            time.time() - started,
        )

        ma_registry = companies[["company_name"]].rename(
            columns={"company_name": "ma_company_name"}
        )
        ma_join_plan = memory_dataset(
            f"{TASK_ID}-ma-events", ma_candidates
        ).sem_join(
            memory_dataset(f"{TASK_ID}-ma-registry", ma_registry),
            condition=(
                "Match only when at least one whole-company merger or acquisition "
                "counterparty name identifies the same legal entity as the registry "
                "company. Allow harmless abbreviations, spelling variants, former "
                "names, and legal-suffix differences, but reject merely similar "
                "unrelated names."
            ),
            depends_on=["ma_counterparty_names", "ma_company_name"],
        )
        started = time.time()
        ma_join_result = ma_join_plan.run(config)
        matched_triples = result_frame(ma_join_result)
        if matched_triples.empty:
            matched_triples = pd.DataFrame(columns=MA_JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(ma_candidates), "right": len(ma_registry)},
            _intermediate_view(matched_triples),
            ma_join_result,
            time.time() - started,
        )

        company_names = (
            matched_triples["company_name"].fillna("").astype(str).str.casefold()
        )
        matched_finance_names = (
            matched_triples["finance_company_name"]
            .fillna("")
            .astype(str)
            .str.casefold()
        )
        ma_names = (
            matched_triples["ma_company_name"]
            .fillna("")
            .astype(str)
            .str.casefold()
        )
        qualifying = matched_triples.loc[
            company_names.ne("")
            & matched_finance_names.ne("")
            & ma_names.ne("")
            & company_names.ne(ma_names)
            & matched_finance_names.ne(ma_names)
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(matched_triples),
            _intermediate_view(qualifying),
        )

        projected = qualifying[
            ["company_name", "ma_company_name", "finance_company_name"]
        ].rename(
            columns={
                "company_name": "company",
                "ma_company_name": "ma_counterparty",
                "finance_company_name": "finance_subsidiary",
            }
        )
        tracker.record("project", len(qualifying), projected)

        answer_frame = (
            projected.assign(
                _company_key=projected["company"].str.casefold(),
                _ma_key=projected["ma_counterparty"].str.casefold(),
                _finance_key=projected["finance_subsidiary"].str.casefold(),
            )
            .drop_duplicates(
                ["_company_key", "_ma_key", "_finance_key"],
                keep="first",
            )
            .drop(columns=["_company_key", "_ma_key", "_finance_key"])
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(projected), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
