#!/usr/bin/env python3
"""Palimpzest pipeline for finance-037."""

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

TASK_ID = "finance-037"
DATASET = "SEC"
GEOGRAPHIC_LEVELS = (
    "domestic_only",
    "limited_international",
    "globally_diversified",
    "undetermined",
)
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


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_2021", "document_2023"],
        errors="ignore",
    )


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
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _foreign_country_count(value) -> int:
    countries = {
        COUNTRY_ALIASES.get(normalized, normalized)
        for item in _string_items(value)
        if (normalized := _normalized_words(item))
    }
    countries -= NON_COUNTRY_VALUES
    countries.discard("united states")
    return len(countries)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings_2021 = load_table(
            DATASET,
            "CSV/2021.csv",
            ["cik", "text", "word_count"],
        )
        filings_2021["cik"] = pd.to_numeric(
            filings_2021["cik"], errors="coerce"
        ).astype("Int64")
        filings_2021["word_count"] = pd.to_numeric(
            filings_2021["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2021)

        candidates_2021 = filings_2021.loc[
            filings_2021["word_count"] > 60_000
        ].reset_index(drop=True)
        tracker.record("filter", len(filings_2021), candidates_2021)

        documents_2021 = load_selected_texts(
            DATASET,
            candidates_2021,
            path_column="text",
            output_column="document_2021",
        )[["cik", "word_count", "document_2021"]]
        tracker.record(
            "scan",
            len(candidates_2021),
            _intermediate_view(documents_2021),
        )

        geography_2021_plan = memory_dataset(
            f"{TASK_ID}-geography-2021", documents_2021
        ).sem_map(
            cols=[
                {
                    "name": "geographic_diversification",
                    "type": str,
                    "desc": (
                        "Exactly one operating-footprint category: domestic_only, "
                        "limited_international, globally_diversified, or "
                        "undetermined."
                    ),
                }
            ],
            desc=(
                "Assign the 2021 filing to its geographic operating-footprint "
                "category."
            ),
            depends_on=["document_2021"],
        )
        started = time.time()
        geography_2021_result = geography_2021_plan.run(config)
        classified_2021 = result_frame(
            geography_2021_result,
            documents_2021,
            ["geographic_diversification"],
        )
        classified_2021["geographic_diversification"] = classified_2021[
            "geographic_diversification"
        ].map(lambda value: normalize_enum(value, GEOGRAPHIC_LEVELS))
        tracker.record_semantic(
            "sem_map",
            len(documents_2021),
            _intermediate_view(classified_2021),
            geography_2021_result,
            time.time() - started,
        )

        domestic_2021 = classified_2021.loc[
            classified_2021["geographic_diversification"].eq("domestic_only")
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(classified_2021),
            _intermediate_view(domestic_2021),
        )

        no_currency_2021_plan = memory_dataset(
            f"{TASK_ID}-no-currency-risk-2021", domestic_2021
        ).sem_filter(
            filter=(
                "Keep the filing only if it does not disclose foreign-currency or "
                "exchange-rate risk."
            ),
            depends_on=["document_2021"],
        )
        started = time.time()
        no_currency_2021_result = no_currency_2021_plan.run(config)
        baseline_candidates = result_frame(
            no_currency_2021_result,
            domestic_2021,
        )
        tracker.record_semantic(
            "sem_filter",
            len(domestic_2021),
            _intermediate_view(baseline_candidates),
            no_currency_2021_result,
            time.time() - started,
        )

        baseline_2021 = (
            baseline_candidates.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .reset_index(drop=True)
        )
        tracker.record(
            "dedup",
            len(baseline_candidates),
            _intermediate_view(baseline_2021),
        )

        filings_2023 = load_table(
            DATASET,
            "CSV/2023.csv",
            ["cik", "text", "word_count"],
        )
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        filings_2023["word_count"] = pd.to_numeric(
            filings_2023["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2023)

        candidates_2023 = filings_2023.loc[
            filings_2023["word_count"] > 60_000
        ].reset_index(drop=True)
        tracker.record("filter", len(filings_2023), candidates_2023)

        documents_2023 = load_selected_texts(
            DATASET,
            candidates_2023,
            path_column="text",
            output_column="document_2023",
        )[["cik", "word_count", "document_2023"]]
        tracker.record(
            "scan",
            len(candidates_2023),
            _intermediate_view(documents_2023),
        )

        geography_2023_plan = memory_dataset(
            f"{TASK_ID}-geography-2023", documents_2023
        ).sem_map(
            cols=[
                {
                    "name": "geographic_diversification",
                    "type": str,
                    "desc": (
                        "Exactly one operating-footprint category: domestic_only, "
                        "limited_international, globally_diversified, or "
                        "undetermined."
                    ),
                }
            ],
            desc=(
                "Assign the 2023 filing to its geographic operating-footprint "
                "category."
            ),
            depends_on=["document_2023"],
        )
        started = time.time()
        geography_2023_result = geography_2023_plan.run(config)
        classified_2023 = result_frame(
            geography_2023_result,
            documents_2023,
            ["geographic_diversification"],
        )
        classified_2023["geographic_diversification"] = classified_2023[
            "geographic_diversification"
        ].map(lambda value: normalize_enum(value, GEOGRAPHIC_LEVELS))
        tracker.record_semantic(
            "sem_map",
            len(documents_2023),
            _intermediate_view(classified_2023),
            geography_2023_result,
            time.time() - started,
        )

        global_2023 = classified_2023.loc[
            classified_2023["geographic_diversification"].eq(
                "globally_diversified"
            )
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(classified_2023),
            _intermediate_view(global_2023),
        )

        currency_2023_plan = memory_dataset(
            f"{TASK_ID}-currency-risk-2023", global_2023
        ).sem_filter(
            filter=(
                "Keep the filing only if it discloses foreign-currency or "
                "exchange-rate risk."
            ),
            depends_on=["document_2023"],
        )
        started = time.time()
        currency_2023_result = currency_2023_plan.run(config)
        expanded_candidates = result_frame(
            currency_2023_result,
            global_2023,
        )
        tracker.record_semantic(
            "sem_filter",
            len(global_2023),
            _intermediate_view(expanded_candidates),
            currency_2023_result,
            time.time() - started,
        )

        expanded_2023 = (
            expanded_candidates.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .reset_index(drop=True)
        )
        tracker.record(
            "dedup",
            len(expanded_candidates),
            _intermediate_view(expanded_2023),
        )

        transitions = baseline_2021.loc[baseline_2021["cik"].notna()].merge(
            expanded_2023.loc[expanded_2023["cik"].notna()],
            on="cik",
            how="inner",
            sort=False,
            suffixes=("_2021", "_2023"),
        )
        tracker.record(
            "join",
            {"left": len(baseline_2021), "right": len(expanded_2023)},
            _intermediate_view(transitions),
        )

        strategy_plan = memory_dataset(
            f"{TASK_ID}-growth-strategy", transitions
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
            desc="Assign the 2023 filing to its stated growth-strategy category.",
            depends_on=["document_2023"],
        )
        started = time.time()
        strategy_result = strategy_plan.run(config)
        growth_labeled = result_frame(
            strategy_result,
            transitions,
            ["growth_strategy"],
        )
        growth_labeled["growth_strategy"] = growth_labeled[
            "growth_strategy"
        ].map(lambda value: normalize_enum(value, GROWTH_STRATEGIES))
        tracker.record_semantic(
            "sem_map",
            len(transitions),
            _intermediate_view(growth_labeled),
            strategy_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-company-details", growth_labeled
        ).sem_map(
            cols=[
                {
                    "name": "company_name",
                    "type": str,
                    "desc": "The reporting company's legal name in the 2023 filing.",
                },
                {
                    "name": "industry_sector",
                    "type": str,
                    "desc": "The reporting company's 2023 industry sector.",
                },
                {
                    "name": "countries_of_operation",
                    "type": list[str],
                    "desc": (
                        "Distinct countries in which the 2023 filing describes "
                        "actual company operations; an empty list when none are "
                        "disclosed."
                    ),
                },
            ],
            desc=(
                "Extract the 2023 company legal name, industry sector, and countries "
                "of operation."
            ),
            depends_on=["document_2023"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            growth_labeled,
            ["company_name", "industry_sector", "countries_of_operation"],
        )
        tracker.record_semantic(
            "sem_map",
            len(growth_labeled),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        projected = extracted[
            ["company_name", "industry_sector", "growth_strategy"]
        ].copy()
        projected["foreign_country_count"] = extracted[
            "countries_of_operation"
        ].map(_foreign_country_count)
        tracker.record("project", len(extracted), projected)

        ordered = projected.assign(
            _company_sort=projected["company_name"].astype(str).str.casefold()
        ).sort_values(
            ["foreign_country_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        )
        tracker.record("order_by", len(projected), ordered)

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), limited)

        answer_frame = limited[
            [
                "company_name",
                "industry_sector",
                "growth_strategy",
                "foreign_country_count",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
