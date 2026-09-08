#!/usr/bin/env python3
"""LOTUS pipeline for finance-036."""

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
    normalize_enum,
    parse_number,
    save_output,
    sem_extract_in_batches,
    sem_map_in_batches,
    setup,
)

TASK_ID = "finance-036"
BATCH_SIZE = 100
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


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _intermediate_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["document_text"], errors="ignore")


def _normalize_entity_type(value) -> str | None:
    normalized = normalize_enum(value, ENTITY_TYPE_LABELS)
    return ENTITY_TYPE_LABELS.get(normalized) if normalized is not None else None


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings = load_table("SEC", "CSV/2020.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings["cik"] = pd.to_numeric(filings["cik"], errors="coerce").astype(
            "Int64"
        )
        filings["word_count"] = pd.to_numeric(
            filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2020.csv)", None, len(filings), output=filings
        )

        candidates = filings[filings["word_count"] > 55000].copy()
        tracker.record(
            "FILTER(word_count > 55000)",
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
            "SEM_FILTER(climate-change risk)",
            input_rows=len(candidates),
        ) as step:
            climate_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                climate_parts.append(
                    documents.sem_filter(
                        "The 2020 10-K filing {document_text} identifies climate "
                        "change as a risk to the reporting company."
                    )
                )
            climate_risk = (
                pd.concat(climate_parts, ignore_index=True)
                if climate_parts
                else _empty_documents(candidates)
            )
            step.set_output(_intermediate_view(climate_risk))

        with tracker.step(
            "SEM_CLASSIFY(entity-type category)",
            input_rows=len(climate_risk),
        ) as step:
            entity_classified = sem_map_in_batches(
                climate_risk,
                "Assign the reporting company's entity type from the 2020 10-K "
                "filing {document_text}. Output exactly one label: "
                "operating_company, holding_company, investment_fund, "
                "blank_check_SPAC, shell_company, limited_partnership, "
                "business_development_company, or undetermined when it cannot be "
                "determined.",
                suffix="entity_type",
                batch_size=BATCH_SIZE,
            )
            entity_classified["entity_type"] = entity_classified[
                "entity_type"
            ].map(_normalize_entity_type)
            step.set_output(_intermediate_view(entity_classified))

        with tracker.step(
            "SEM_CLASSIFY(stated growth-strategy category)",
            input_rows=len(entity_classified),
        ) as step:
            classified = sem_map_in_batches(
                entity_classified,
                "Assign the reporting company's stated growth strategy from the "
                "2020 10-K filing {document_text}. Output exactly one label: "
                "organic for internal business growth; acquisition_driven for "
                "growth primarily through acquisitions; hybrid when both are stated "
                "components; not_applicable when no growth strategy applies; or "
                "undetermined when it cannot be determined.",
                suffix="growth_strategy",
                batch_size=BATCH_SIZE,
            )
            classified["growth_strategy"] = classified["growth_strategy"].map(
                lambda value: normalize_enum(value, GROWTH_STRATEGIES)
            )
            step.set_output(_intermediate_view(classified))

        with tracker.step(
            "SEM_EXTRACT(company, industry, and total risk-factor count)",
            input_rows=len(classified),
        ) as step:
            extracted = sem_extract_in_batches(
                classified,
                input_cols=["document_text"],
                output_cols={
                    "company_name": (
                        "the reporting company's legal name stated in the filing"
                    ),
                    "industry_sector": (
                        "the reporting company's industry sector stated or clearly "
                        "supported by the filing"
                    ),
                    "risk_factor_count": (
                        "the total number of distinct risk factors in the filing's "
                        "Risk Factors section as an integer, or null when the total "
                        "cannot be determined"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted["company_name"] = extracted["company_name"].map(clean_text)
            extracted["industry_sector"] = extracted["industry_sector"].map(
                clean_text
            )
            extracted["risk_factor_count"] = pd.to_numeric(
                extracted["risk_factor_count"].map(parse_number), errors="coerce"
            )
            step.set_output(_intermediate_view(extracted))

        qualifying = extracted[
            extracted["entity_type"].notna()
            & (extracted["entity_type"] != "undetermined")
            & extracted["growth_strategy"].notna()
            & (extracted["growth_strategy"] != "undetermined")
            & extracted["risk_factor_count"].notna()
        ].copy()
        tracker.record(
            "FILTER(entity_type != 'undetermined' AND growth_strategy != 'undetermined' AND risk_factor_count IS NOT NULL)",
            len(extracted),
            len(qualifying),
            output=_intermediate_view(qualifying),
        )

        deduped = (
            qualifying.sort_values(
                ["company_name", "risk_factor_count", "word_count", "cik"],
                ascending=[True, False, False, False],
                kind="mergesort",
            )
            .drop_duplicates("company_name", keep="first")
            .copy()
        )
        tracker.record(
            "DEDUP(company_name, keep=max_by(risk_factor_count, word_count, cik))",
            len(qualifying),
            len(deduped),
            output=_intermediate_view(deduped),
        )

        ranked = deduped.assign(
            _company_sort=deduped["company_name"].str.lower()
        ).sort_values(
            ["risk_factor_count", "_company_sort"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)
        answer_frame = ranked[
            [
                "company_name",
                "industry_sector",
                "entity_type",
                "growth_strategy",
                "risk_factor_count",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "ORDER_BY(risk_factor_count DESC, LOWER(company_name) ASC) -> LIMIT(10) -> PROJECT",
            len(deduped),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
