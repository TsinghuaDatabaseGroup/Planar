#!/usr/bin/env python3
"""LOTUS pipeline for finance-030."""

import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    iter_selected_texts,
    load_table,
    normalize_enum,
    parse_string_list,
    save_output,
    sem_extract_in_batches,
    sem_filter_in_batches,
    sem_map_in_batches,
    setup,
)

TASK_ID = "finance-030"
BATCH_SIZE = 100
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


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _intermediate_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["document_text"], errors="ignore")


def _normalized_words(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
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
    countries = {_normalize_country(item) for item in parse_string_list(value)}
    countries -= NON_COUNTRY_VALUES
    countries.discard("united states")
    return len(countries)


def _subsidiary_count(value) -> int:
    subsidiaries = {_normalize_entity(item) for item in parse_string_list(value)}
    subsidiaries.discard("")
    return len(subsidiaries)


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2022.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2022.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 60000].copy()
        tracker.record(
            "FILTER(word_count > 60000)",
            len(filings),
            len(candidates),
            output=candidates,
        )
        tracker.record(
            "SCAN_DOCS(selector=filtered.text)",
            len(candidates),
            len(candidates),
            output=candidates,
        )

        with tracker.step(
            "SEM_FILTER(dependence on a major customer)",
            input_rows=len(candidates),
        ) as step:
            customer_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                customer_parts.append(
                    documents.sem_filter(
                        "The 10-K filing {document_text} discloses dependence on a "
                        "major customer or material customer concentration for the "
                        "reporting company."
                    )
                )
            customer_filings = (
                pd.concat(customer_parts, ignore_index=True)
                if customer_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(customer_filings))

        with tracker.step(
            "SEM_FILTER(foreign-currency or exchange-rate risk)",
            input_rows=len(customer_filings),
        ) as step:
            currency_filings = sem_filter_in_batches(
                customer_filings,
                "The 10-K filing {document_text} discloses foreign-currency or "
                "exchange-rate risk for the reporting company.",
                batch_size=BATCH_SIZE,
            )
            step.set_output(_intermediate_view(currency_filings))

        with tracker.step(
            "SEM_CLASSIFY(stated growth-strategy category)",
            input_rows=len(currency_filings),
        ) as step:
            classified = sem_map_in_batches(
                currency_filings,
                "Classify the reporting company's stated growth strategy in the "
                "10-K filing {document_text}. Output exactly one label: organic for "
                "growth through internal business activities; acquisition_driven "
                "for growth primarily through acquisitions; hybrid when both "
                "organic growth and acquisitions are stated components; "
                "not_applicable when no growth strategy applies; or undetermined "
                "when it cannot be determined.",
                suffix="growth_strategy",
                batch_size=BATCH_SIZE,
            )
            classified["growth_strategy"] = classified["growth_strategy"].map(
                lambda value: normalize_enum(value, GROWTH_STRATEGIES)
            )
            step.set_output(_intermediate_view(classified))

        growth_filings = classified[
            classified["growth_strategy"].isin(["acquisition_driven", "hybrid"])
        ].copy()
        tracker.record(
            "FILTER(growth_strategy IN ['acquisition_driven', 'hybrid'])",
            len(classified),
            len(growth_filings),
            output=_intermediate_view(growth_filings),
        )

        with tracker.step(
            "SEM_EXTRACT(company, industry, countries, and subsidiaries)",
            input_rows=len(growth_filings),
        ) as step:
            extracted = sem_extract_in_batches(
                growth_filings,
                input_cols=["document_text"],
                output_cols={
                    "company_name": (
                        "the reporting company's legal name stated in the filing"
                    ),
                    "industry_sector": (
                        "a concise industry sector stated or clearly supported by "
                        "the filing"
                    ),
                    "countries_of_operation": (
                        "a JSON list of every distinct country in which the filing "
                        "describes actual company operations; use an empty list when "
                        "none are disclosed"
                    ),
                    "subsidiary_names": (
                        "a JSON list of every distinct specifically named legal "
                        "subsidiary disclosed in the filing; do not include the "
                        "registrant itself, and use an empty list when none are disclosed"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted["company_name"] = extracted["company_name"].map(clean_text)
            extracted["industry_sector"] = extracted["industry_sector"].map(
                clean_text
            )
            step.set_output(_intermediate_view(extracted))

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
        tracker.record(
            "PROJECT(normalized foreign-country and subsidiary counts)",
            len(extracted),
            len(projected),
            output=projected,
        )

        foreign = projected[projected["country_count"] > 0].copy()
        tracker.record(
            "FILTER(country_count > 0)",
            len(projected),
            len(foreign),
            output=foreign,
        )

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
            .copy()
        )
        tracker.record(
            "DEDUP(company_name, keep=max_by(country_count, subsidiary_count, word_count, cik))",
            len(foreign),
            len(deduped),
            output=deduped,
        )

        ranked = deduped.assign(
            _company_sort=deduped["company_name"].str.lower()
        ).sort_values(
            ["country_count", "subsidiary_count", "_company_sort"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(10)
        answer_frame = ranked[
            [
                "company_name",
                "industry_sector",
                "growth_strategy",
                "country_count",
                "subsidiary_count",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(country_count DESC, subsidiary_count DESC, LOWER(company_name) ASC) -> LIMIT(10) -> PROJECT",
            len(deduped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
