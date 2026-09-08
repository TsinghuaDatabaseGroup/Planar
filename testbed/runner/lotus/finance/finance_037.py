#!/usr/bin/env python3
"""LOTUS pipeline for finance-037."""

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
    sem_map_in_batches,
    setup,
)

TASK_ID = "finance-037"
BATCH_SIZE = 100
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


def _normalized_words(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _foreign_country_count(value) -> int:
    countries = {
        COUNTRY_ALIASES.get(normalized, normalized)
        for item in parse_string_list(value)
        if (normalized := _normalized_words(item))
    }
    countries -= NON_COUNTRY_VALUES
    countries.discard("united states")
    return len(countries)


def _classify_geography(
    filings: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    parts = []
    for documents in iter_selected_texts(
        "SEC",
        filings,
        path_column="text",
        output_column=f"document_{year}",
        batch_size=BATCH_SIZE,
    ):
        batch = documents.sem_map(
            f"Assign the reporting company's geographic operating footprint from "
            f"the {year} 10-K filing {{document_{year}}}. Output exactly one label: "
            "domestic_only, limited_international, globally_diversified, or "
            "undetermined when it cannot be determined.",
            suffix="geographic_diversification",
        )
        batch["geographic_diversification"] = batch[
            "geographic_diversification"
        ].map(lambda value: normalize_enum(value, GEOGRAPHIC_LEVELS))
        parts.append(batch.drop(columns=[f"document_{year}"], errors="ignore"))
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else filings.iloc[0:0].assign(
            geographic_diversification=pd.Series(dtype="object")
        )
    )


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings_2021 = load_table("SEC", "CSV/2021.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings_2021["cik"] = pd.to_numeric(
            filings_2021["cik"], errors="coerce"
        ).astype("Int64")
        filings_2021["word_count"] = pd.to_numeric(
            filings_2021["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2021.csv AS filings_2021)",
            None,
            len(filings_2021),
            output=filings_2021,
        )
        candidates_2021 = filings_2021[filings_2021["word_count"] > 60000].copy()
        tracker.record(
            "FILTER(2021 word_count > 60000)",
            len(filings_2021),
            len(candidates_2021),
            output=candidates_2021,
        )
        tracker.record(
            "SCAN_DOCS(selector=2021 filtered.text)",
            len(candidates_2021),
            len(candidates_2021),
            output=candidates_2021,
        )

        with tracker.step(
            "SEM_CLASSIFY(2021 geographic operating footprint)",
            input_rows=len(candidates_2021),
        ) as step:
            classified_2021 = _classify_geography(candidates_2021, 2021)
            step.set_output(classified_2021)

        domestic_2021 = classified_2021[
            classified_2021["geographic_diversification"] == "domestic_only"
        ].copy()
        tracker.record(
            "FILTER(2021 geographic_diversification = 'domestic_only')",
            len(classified_2021),
            len(domestic_2021),
            output=domestic_2021,
        )

        with tracker.step(
            "SEM_FILTER(2021 does not disclose foreign-currency risk)",
            input_rows=len(domestic_2021),
        ) as step:
            baseline_parts = []
            for documents in iter_selected_texts(
                "SEC",
                domestic_2021,
                path_column="text",
                output_column="document_2021",
                batch_size=BATCH_SIZE,
            ):
                selected = documents.sem_filter(
                    "The 2021 10-K filing {document_2021} does not disclose foreign-"
                    "currency or exchange-rate risk for the reporting company."
                )
                baseline_parts.append(
                    selected.drop(columns=["document_2021"], errors="ignore")
                )
            baseline_candidates = (
                pd.concat(baseline_parts, ignore_index=True)
                if baseline_parts
                else domestic_2021.iloc[0:0].copy()
            )
            step.set_output(baseline_candidates)

        baseline_2021 = (
            baseline_candidates.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .copy()
        )
        tracker.record(
            "DEDUP(2021 cik, keep=max_by(word_count))",
            len(baseline_candidates),
            len(baseline_2021),
            output=baseline_2021,
        )

        filings_2023 = load_table("SEC", "CSV/2023.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings_2023["cik"] = pd.to_numeric(
            filings_2023["cik"], errors="coerce"
        ).astype("Int64")
        filings_2023["word_count"] = pd.to_numeric(
            filings_2023["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2023.csv AS filings_2023)",
            None,
            len(filings_2023),
            output=filings_2023,
        )
        candidates_2023 = filings_2023[filings_2023["word_count"] > 60000].copy()
        tracker.record(
            "FILTER(2023 word_count > 60000)",
            len(filings_2023),
            len(candidates_2023),
            output=candidates_2023,
        )
        tracker.record(
            "SCAN_DOCS(selector=2023 filtered.text)",
            len(candidates_2023),
            len(candidates_2023),
            output=candidates_2023,
        )

        with tracker.step(
            "SEM_CLASSIFY(2023 geographic operating footprint)",
            input_rows=len(candidates_2023),
        ) as step:
            classified_2023 = _classify_geography(candidates_2023, 2023)
            step.set_output(classified_2023)

        global_2023 = classified_2023[
            classified_2023["geographic_diversification"]
            == "globally_diversified"
        ].copy()
        tracker.record(
            "FILTER(2023 geographic_diversification = 'globally_diversified')",
            len(classified_2023),
            len(global_2023),
            output=global_2023,
        )

        with tracker.step(
            "SEM_FILTER(2023 discloses foreign-currency risk)",
            input_rows=len(global_2023),
        ) as step:
            expanded_parts = []
            for documents in iter_selected_texts(
                "SEC",
                global_2023,
                path_column="text",
                output_column="document_2023",
                batch_size=BATCH_SIZE,
            ):
                expanded_parts.append(
                    documents.sem_filter(
                        "The 2023 10-K filing {document_2023} discloses foreign-"
                        "currency or exchange-rate risk for the reporting company."
                    )
                )
            expanded_candidates = (
                pd.concat(expanded_parts, ignore_index=True)
                if expanded_parts
                else global_2023.iloc[0:0].assign(
                    document_2023=pd.Series(dtype="object")
                )
            )
            step.set_output(
                expanded_candidates.drop(columns=["document_2023"], errors="ignore")
            )

        expanded_2023 = (
            expanded_candidates.sort_values(
                ["cik", "word_count"],
                ascending=[True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .copy()
        )
        tracker.record(
            "DEDUP(2023 cik, keep=max_by(word_count))",
            len(expanded_candidates),
            len(expanded_2023),
            output=expanded_2023.drop(columns=["document_2023"], errors="ignore"),
        )

        transitions = baseline_2021.merge(
            expanded_2023,
            on="cik",
            how="inner",
            sort=False,
            suffixes=("_2021", "_2023"),
        )
        tracker.record(
            "JOIN(inner, baseline_2021.cik = expanded_2023.cik)",
            {"left": len(baseline_2021), "right": len(expanded_2023)},
            len(transitions),
            output=transitions.drop(columns=["document_2023"], errors="ignore"),
        )

        with tracker.step(
            "SEM_CLASSIFY(2023 stated growth-strategy category)",
            input_rows=len(transitions),
        ) as step:
            growth_labeled = sem_map_in_batches(
                transitions,
                "Assign the reporting company's stated growth strategy from the "
                "2023 10-K filing {document_2023}. Output exactly one label: "
                "organic, acquisition_driven, hybrid, not_applicable, or "
                "undetermined when it cannot be determined.",
                suffix="growth_strategy",
                batch_size=BATCH_SIZE,
            )
            growth_labeled["growth_strategy"] = growth_labeled[
                "growth_strategy"
            ].map(lambda value: normalize_enum(value, GROWTH_STRATEGIES))
            step.set_output(
                growth_labeled.drop(columns=["document_2023"], errors="ignore")
            )

        with tracker.step(
            "SEM_EXTRACT(2023 company, industry, and countries of operation)",
            input_rows=len(growth_labeled),
        ) as step:
            extracted = sem_extract_in_batches(
                growth_labeled,
                input_cols=["document_2023"],
                output_cols={
                    "company_name": (
                        "the reporting company's legal name stated in the 2023 filing"
                    ),
                    "industry_sector": (
                        "the reporting company's industry sector stated or clearly "
                        "supported by the 2023 filing"
                    ),
                    "countries_of_operation": (
                        "a JSON list of every distinct country in which the 2023 "
                        "filing describes actual company operations; use an empty "
                        "list when none are disclosed"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted["company_name"] = extracted["company_name"].map(clean_text)
            extracted["industry_sector"] = extracted["industry_sector"].map(
                clean_text
            )
            step.set_output(extracted.drop(columns=["document_2023"], errors="ignore"))

        projected = extracted[
            ["company_name", "industry_sector", "growth_strategy"]
        ].copy()
        projected["foreign_country_count"] = extracted[
            "countries_of_operation"
        ].map(_foreign_country_count)
        tracker.record(
            "PROJECT(company, industry, growth strategy, normalized foreign-country count)",
            len(extracted),
            len(projected),
            output=projected,
        )

        answer_frame = projected.assign(
            _company_sort=projected["company_name"].str.lower()
        ).sort_values(
            ["foreign_country_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)[
            [
                "company_name",
                "industry_sector",
                "growth_strategy",
                "foreign_country_count",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(foreign_country_count DESC, LOWER(company_name) ASC) -> LIMIT(10) -> PROJECT",
            len(projected),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
