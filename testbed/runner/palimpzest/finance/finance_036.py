#!/usr/bin/env python3
"""Palimpzest pipeline for finance-036."""

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
    normalize_enum,
    result_frame,
    save_output,
)

TASK_ID = "finance-036"
DATASET = "SEC"
ENTITY_TYPE_LABELS = {
    "operating_company": "operating_company",
    "holding_company": "holding_company",
    "investment_fund": "investment_fund",
    "blank_check_spac": "blank_check_SPAC",
    "shell_company": "shell_company",
    "limited_partnership": "limited_partnership",
    "business_development_company": "business_development_company",
    "undetermined": "undetermined",
}
GROWTH_STRATEGIES = (
    "organic",
    "acquisition_driven",
    "hybrid",
    "not_applicable",
    "undetermined",
)


def _intermediate_view(frame):
    return frame.drop(columns=["document_text"], errors="ignore")


def _normalize_entity_type(value) -> str | None:
    normalized = normalize_enum(value, ENTITY_TYPE_LABELS)
    return ENTITY_TYPE_LABELS.get(normalized) if normalized is not None else None


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings = load_table(
            DATASET,
            "CSV/2020.csv",
            ["cik", "text", "word_count"],
        )
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings)

        candidates = filings.loc[filings["word_count"] > 55_000].reset_index(
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

        climate_plan = memory_dataset(
            f"{TASK_ID}-climate-risk", documents
        ).sem_filter(
            filter=(
                "Keep the filing only if it identifies climate change as a risk "
                "to the reporting company."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        climate_result = climate_plan.run(config)
        climate_risk = result_frame(climate_result, documents)
        tracker.record_semantic(
            "sem_filter",
            len(documents),
            _intermediate_view(climate_risk),
            climate_result,
            time.time() - started,
        )

        entity_plan = memory_dataset(
            f"{TASK_ID}-entity-type", climate_risk
        ).sem_map(
            cols=[
                {
                    "name": "entity_type",
                    "type": str,
                    "desc": (
                        "Exactly one category: operating_company, holding_company, "
                        "investment_fund, blank_check_SPAC, shell_company, "
                        "limited_partnership, business_development_company, or "
                        "undetermined when it cannot be determined."
                    ),
                }
            ],
            desc="Assign the reporting company's entity-type category.",
            depends_on=["document_text"],
        )
        started = time.time()
        entity_result = entity_plan.run(config)
        entity_classified = result_frame(
            entity_result,
            climate_risk,
            ["entity_type"],
        )
        entity_classified["entity_type"] = entity_classified["entity_type"].map(
            _normalize_entity_type
        )
        tracker.record_semantic(
            "sem_map",
            len(climate_risk),
            _intermediate_view(entity_classified),
            entity_result,
            time.time() - started,
        )

        strategy_plan = memory_dataset(
            f"{TASK_ID}-growth-strategy", entity_classified
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
            entity_classified,
            ["growth_strategy"],
        )
        classified["growth_strategy"] = classified["growth_strategy"].map(
            lambda value: normalize_enum(value, GROWTH_STRATEGIES)
        )
        tracker.record_semantic(
            "sem_map",
            len(entity_classified),
            _intermediate_view(classified),
            strategy_result,
            time.time() - started,
        )

        extraction_plan = memory_dataset(
            f"{TASK_ID}-company-details", classified
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
                    "desc": "The reporting company's industry sector.",
                },
                {
                    "name": "risk_factor_count",
                    "type": int,
                    "desc": (
                        "The total number of distinct risk factors in the filing's "
                        "Risk Factors section, or null when the total cannot be "
                        "determined."
                    ),
                },
            ],
            desc=(
                "Extract the company legal name, industry sector, and total number "
                "of risk factors."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        extraction_result = extraction_plan.run(config)
        extracted = result_frame(
            extraction_result,
            classified,
            ["company_name", "industry_sector", "risk_factor_count"],
        )
        extracted["risk_factor_count"] = pd.to_numeric(
            extracted["risk_factor_count"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(classified),
            _intermediate_view(extracted),
            extraction_result,
            time.time() - started,
        )

        qualifying = extracted.loc[
            extracted["entity_type"].notna()
            & extracted["entity_type"].ne("undetermined")
            & extracted["growth_strategy"].notna()
            & extracted["growth_strategy"].ne("undetermined")
            & extracted["risk_factor_count"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(extracted),
            _intermediate_view(qualifying),
        )

        deduped = (
            qualifying.sort_values(
                ["company_name", "risk_factor_count", "word_count", "cik"],
                ascending=[True, False, False, False],
                kind="mergesort",
            )
            .drop_duplicates("company_name", keep="first")
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(qualifying), _intermediate_view(deduped))

        ordered = deduped.assign(
            _company_sort=deduped["company_name"].astype(str).str.casefold()
        ).sort_values(
            ["risk_factor_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        )
        tracker.record("order_by", len(deduped), _intermediate_view(ordered))

        limited = ordered.head(10).reset_index(drop=True)
        tracker.record("limit", len(ordered), _intermediate_view(limited))

        answer_frame = limited[
            [
                "company_name",
                "industry_sector",
                "entity_type",
                "growth_strategy",
                "risk_factor_count",
            ]
        ].reset_index(drop=True)
        tracker.record("project", len(limited), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
