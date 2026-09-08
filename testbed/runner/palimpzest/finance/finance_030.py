#!/usr/bin/env python3
"""Palimpzest pipeline for finance-030."""

from __future__ import annotations

import os
import re
import sys
import time
import unicodedata

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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "finance-030"
DATASET = "SEC"
GROWTH_STRATEGIES = (
    "organic",
    "acquisition_driven",
    "hybrid",
    "not_applicable",
    "undetermined",
)
COUNTRY_ALIASES = {
    "u s": "united states",
    "u s a": "united states",
    "us": "united states",
    "usa": "united states",
    "united states of america": "united states",
    "america": "united states",
    "u k": "united kingdom",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "people s republic of china": "china",
    "prc": "china",
    "republic of korea": "south korea",
    "korea republic of": "south korea",
    "russian federation": "russia",
    "united arab emirates": "uae",
    "u a e": "uae",
    "czech republic": "czechia",
}
NON_COUNTRY_VALUES = {
    "",
    "global",
    "international",
    "worldwide",
    "various",
    "other",
    "not disclosed",
    "europe",
    "asia",
    "africa",
    "north america",
    "south america",
    "latin america",
    "middle east",
    "asia pacific",
}
ENTITY_SUFFIXES = {
    "ag",
    "bv",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "lp",
    "ltd",
    "nv",
    "plc",
    "pte",
    "sa",
    "sarl",
}


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def _string_items(value) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str) and value.strip():
        values = [value]
    else:
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _normalized_words(value) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _normalize_country(value) -> str:
    country = _normalized_words(value)
    return COUNTRY_ALIASES.get(country, country)


def _normalize_entity(value) -> str:
    tokens = _normalized_words(value).split()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    while tokens and tokens[-1] in ENTITY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _foreign_country_count(value) -> int:
    countries = {_normalize_country(item) for item in _string_items(value)}
    countries -= NON_COUNTRY_VALUES
    countries.discard("united states")
    return len(countries)


def _subsidiary_count(value) -> int:
    subsidiaries = {_normalize_entity(item) for item in _string_items(value)}
    subsidiaries.discard("")
    return len(subsidiaries)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2022.csv",
            ["cik", "text", "word_count"],
        )
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[filings["word_count"] > 60_000].reset_index(
            drop=True
        )
        tracker.record("filter", len(filings), candidates)

        documents = load_selected_texts(
            DATASET,
            candidates,
            path_column="text",
            output_column="document_text",
        )[["cik", "word_count", "document_text"]]
        tracker.record("scan", len(candidates), _intermediate_view(documents))

        customer_plan = memory_dataset(
            f"{TASK_ID}-major-customer", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it discloses dependence on a major "
                "customer or material customer concentration."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        customer_result = customer_plan.run(config)
        customer_filings = result_frame(customer_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(customer_filings),
            customer_result,
            time.time() - started,
        )

        currency_plan = memory_dataset(
            f"{TASK_ID}-currency-risk", customer_filings
        ).sem_filter(
            filter=(
                "Keep the filing only if it discloses foreign-currency or "
                "exchange-rate risk."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        currency_result = currency_plan.run(config)
        currency_filings = result_frame(currency_result, customer_filings)
        tracker.record_semantic(
            "sem_filter",
            len(customer_filings),
            _intermediate_view(currency_filings),
            currency_result,
            time.time() - started,
        )

        strategy_plan = memory_dataset(
            f"{TASK_ID}-growth-strategy", currency_filings
        ).sem_map(
            cols=[
                {
                    "name": "growth_strategy",
                    "type": str,
                    "desc": (
                        "Exactly one stated growth-strategy category: organic, "
                        "acquisition_driven, hybrid, not_applicable, or undetermined."
                    ),
                }
            ],
            desc="Assign the filing to its stated growth-strategy category.",
            depends_on=["document_text"],
        )
        started = time.time()
        strategy_result = strategy_plan.run(config)
        classified = result_frame(
            strategy_result,
            currency_filings,
            ["growth_strategy"],
        )
        classified["growth_strategy"] = classified["growth_strategy"].map(
            lambda value: normalize_enum(value, GROWTH_STRATEGIES)
        )
        tracker.record_semantic(
            "sem_map",
            len(currency_filings),
            _intermediate_view(classified),
            strategy_result,
            time.time() - started,
        )

        growth_filings = classified.loc[
            classified["growth_strategy"].isin(
                ["acquisition_driven", "hybrid"]
            )
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(classified),
            _intermediate_view(growth_filings),
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-details", growth_filings
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The reporting company's legal name.",
                },
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": "The reporting company's primary industry sector.",
                },
                {
                    "name": "countries_of_operation",
                    "type": list[str],
                    "desc": (
                        "Distinct countries in which the filing describes actual "
                        "company operations; an empty list when none are disclosed."
                    ),
                },
                {
                    "name": "subsidiary_names",
                    "type": list[str],
                    "desc": (
                        "Distinct specifically named legal subsidiaries disclosed "
                        "in the filing, excluding the registrant itself; an empty "
                        "list when none are disclosed."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name, industry sector, countries of "
                "operation, and disclosed subsidiary names."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            growth_filings,
            [
                "company_name",
                "industry_sector",
                "countries_of_operation",
                "subsidiary_names",
            ],
        )
        tracker.record_semantic(
            "sem_map",
            len(growth_filings),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        projected = extracted[
            [
                "cik",
                "word_count",
                "company_name",
                "industry_sector",
                "growth_strategy",
            ]
        ].copy()
        projected["country_count"] = extracted["countries_of_operation"].map(
            _foreign_country_count
        )
        projected["subsidiary_count"] = extracted["subsidiary_names"].map(
            _subsidiary_count
        )
        tracker.record("project", len(extracted), projected)

        foreign = projected.loc[projected["country_count"] > 0].reset_index(
            drop=True
        )
        tracker.record("filter", len(projected), foreign)

        deduped = (
            foreign.sort_values(
                [
                    "company_name",
                    "country_count",
                    "subsidiary_count",
                    "word_count",
                    "cik",
                ],
                ascending=[True, False, False, False, False],
                kind="mergesort",
            )
            .drop_duplicates("company_name", keep="first")
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(foreign), deduped)

        ordered = deduped.assign(
            _company_sort=deduped["company_name"].astype(str).str.casefold()
        ).sort_values(
            ["country_count", "subsidiary_count", "_company_sort"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        tracker.record("order_by", len(deduped), ordered)

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited[
            [
                "company_name",
                "industry_sector",
                "growth_strategy",
                "country_count",
                "subsidiary_count",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
