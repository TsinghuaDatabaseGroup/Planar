#!/usr/bin/env python3
"""LOTUS pipeline for finance-042."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    iter_selected_texts,
    load_table,
    parse_string_list,
    save_output,
    sem_extract_in_batches,
    setup,
)

TASK_ID = "finance-042"
BATCH_SIZE = 100
YEARS = tuple(range(2019, 2025))


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _without_large_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            "document_text",
            "filing_texts",
            "finance_join_binding",
            "finance_registry_join_binding",
            "ma_join_binding",
            "ma_registry_join_binding",
        ],
        errors="ignore",
    )


def _union_distinct_names(values) -> list[str]:
    names = []
    seen = set()
    for value in values:
        for raw_name in parse_string_list(value):
            name = clean_text(raw_name, default="")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = pd.concat(
            [
                load_table("SEC", f"CSV/{year}.csv")[["text"]].copy()
                for year in YEARS
            ],
            ignore_index=True,
        )
        tracker.record(
            "SCAN_TABLE(CSV)",
            None,
            len(filings),
            output=filings,
        )
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_EXTRACT(company and wholly-owned finance subsidiaries)",
            input_rows=len(filings),
        ) as step:
            filing_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["document_text"],
                    output_cols={
                        "company_name": (
                            "the filing company's legal registrant name"
                        ),
                        "finance_subsidiary_names": (
                            "a JSON list of distinct legal names of entities that "
                            "the filing explicitly identifies as wholly-owned "
                            "finance or funding subsidiaries of the reporting "
                            "company; use an empty list when none are disclosed"
                        ),
                    },
                )
                batch["company_name"] = batch["company_name"].map(clean_text)
                batch["finance_subsidiary_names"] = batch[
                    "finance_subsidiary_names"
                ].map(parse_string_list)
                filing_parts.append(batch)
            extracted_filings = (
                pd.concat(filing_parts, ignore_index=True)
                if filing_parts
                else _empty_documents(filings).assign(
                    company_name=pd.Series(dtype="object"),
                    finance_subsidiary_names=pd.Series(dtype="object"),
                )
            )
            step.set_output(_without_large_columns(extracted_filings))

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
            "GROUP_BY(company_name, UNION_DISTINCT(finance names), COLLECT(filings))",
            len(extracted_filings),
            len(companies),
            output=_without_large_columns(companies),
        )

        finance_candidates = companies[
            companies["finance_subsidiary_names"].map(len) > 0
        ].copy()
        tracker.record(
            "FILTER(LENGTH(finance_subsidiary_names) > 0)",
            len(companies),
            len(finance_candidates),
            output=_without_large_columns(finance_candidates),
        )

        finance_left = finance_candidates.copy()
        finance_left["finance_join_binding"] = finance_left[
            "finance_subsidiary_names"
        ].map(lambda names: json.dumps(names, ensure_ascii=False))
        finance_registry = companies[["company_name"]].rename(
            columns={"company_name": "finance_company_name"}
        )
        finance_registry["finance_registry_join_binding"] = finance_registry[
            "finance_company_name"
        ].astype(str)
        tracker.record(
            "CODE_MAP(bind finance-subsidiary SEM_JOIN inputs)",
            len(finance_candidates),
            len(finance_left),
            output=_without_large_columns(finance_left),
        )
        tracker.record(
            "CODE_MAP(bind finance registry SEM_JOIN inputs)",
            len(companies),
            len(finance_registry),
            output=_without_large_columns(finance_registry),
        )

        with tracker.step(
            "SEM_JOIN(finance-subsidiary name to same filer legal entity)",
            input_rows={"left": len(finance_left), "right": len(finance_registry)},
        ) as step:
            if finance_left.empty or finance_registry.empty:
                finance_matches = pd.DataFrame(
                    columns=[*finance_left.columns, *finance_registry.columns]
                )
            else:
                finance_matches = finance_left.sem_join(
                    finance_registry,
                    "Match at least one wholly-owned finance or funding subsidiary "
                    "name in {finance_join_binding} to "
                    "{finance_registry_join_binding} only when they identify the "
                    "same legal entity despite harmless abbreviations, spelling "
                    "variants, former names, or legal-suffix differences."
                )
            step.set_output(_without_large_columns(finance_matches))

        owner_names = (
            finance_matches["company_name"].fillna("").astype(str).str.casefold()
        )
        finance_names = (
            finance_matches["finance_company_name"]
            .fillna("")
            .astype(str)
            .str.casefold()
        )
        distinct_finance_matches = finance_matches[
            owner_names.ne("")
            & finance_names.ne("")
            & owner_names.ne(finance_names)
        ].copy()
        tracker.record(
            "FILTER(company_name != finance_company_name)",
            len(finance_matches),
            len(distinct_finance_matches),
            output=_without_large_columns(distinct_finance_matches),
        )

        with tracker.step(
            "SEM_EXTRACT(whole-company M&A counterparties from grouped filings)",
            input_rows=len(distinct_finance_matches),
        ) as step:
            ma_candidates = sem_extract_in_batches(
                distinct_finance_matches,
                input_cols=["filing_texts"],
                output_cols={
                    "ma_counterparty_names": (
                        "a JSON list of distinct legal or commonly stated names of "
                        "whole-company merger or acquisition counterparties "
                        "involving the reporting company across these filings; "
                        "exclude asset purchases, minority-stake investments, "
                        "spin-offs, and SPAC or blank-check business combinations; "
                        "use an empty list when none qualify"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            ma_candidates["ma_counterparty_names"] = ma_candidates[
                "ma_counterparty_names"
            ].map(parse_string_list)
            step.set_output(_without_large_columns(ma_candidates))

        ma_left = ma_candidates[
            ["company_name", "finance_company_name", "ma_counterparty_names"]
        ].copy()
        ma_left["ma_join_binding"] = ma_left["ma_counterparty_names"].map(
            lambda names: json.dumps(names, ensure_ascii=False)
        )
        ma_registry = companies[["company_name"]].rename(
            columns={"company_name": "ma_company_name"}
        )
        ma_registry["ma_registry_join_binding"] = ma_registry[
            "ma_company_name"
        ].astype(str)
        tracker.record(
            "CODE_MAP(bind M&A-counterparty SEM_JOIN inputs)",
            len(ma_candidates),
            len(ma_left),
            output=_without_large_columns(ma_left),
        )
        tracker.record(
            "CODE_MAP(bind M&A registry SEM_JOIN inputs)",
            len(companies),
            len(ma_registry),
            output=_without_large_columns(ma_registry),
        )

        with tracker.step(
            "SEM_JOIN(M&A-counterparty name to same filer legal entity)",
            input_rows={"left": len(ma_left), "right": len(ma_registry)},
        ) as step:
            if ma_left.empty or ma_registry.empty:
                matched_triples = pd.DataFrame(
                    columns=[*ma_left.columns, *ma_registry.columns]
                )
            else:
                matched_triples = ma_left.sem_join(
                    ma_registry,
                    "Match at least one whole-company M&A counterparty name in "
                    "{ma_join_binding} to {ma_registry_join_binding} only when "
                    "they identify the same legal entity despite harmless "
                    "abbreviations, spelling variants, former names, or "
                    "legal-suffix differences."
                )
            step.set_output(_without_large_columns(matched_triples))

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
        qualifying = matched_triples[
            company_names.ne("")
            & matched_finance_names.ne("")
            & ma_names.ne("")
            & company_names.ne(ma_names)
            & matched_finance_names.ne(ma_names)
        ].copy()
        tracker.record(
            "FILTER(company != M&A company AND finance company != M&A company)",
            len(matched_triples),
            len(qualifying),
            output=_without_large_columns(qualifying),
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
        tracker.record(
            "PROJECT(company, ma_counterparty, finance_subsidiary)",
            len(qualifying),
            len(projected),
            output=projected,
        )

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
        tracker.record(
            "DEDUP(company, ma_counterparty, finance_subsidiary)",
            len(projected),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
