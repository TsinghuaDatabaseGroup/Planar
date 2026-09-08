#!/usr/bin/env python3
"""Palimpzest pipeline for finance-039."""

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
    normalize_text_value,
    result_frame,
    save_output,
)

TASK_ID = "finance-039"
DATASET = "SEC"
JOIN_COLUMNS = [
    "acquirer_cik",
    "acquirer",
    "acquirer_sector",
    "acquired_company_names",
    "document_2020",
    "registry_cik",
    "registry_company_name",
    "target_sector",
    "target_risk_factor_count",
    "document_2019",
]


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_2020", "document_2019"],
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


def _clean_nullable_text(value) -> str | None:
    return normalize_text_value(value)


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        filings_2020 = load_table(
            DATASET,
            "CSV/2020.csv",
            ["cik", "text", "word_count"],
        )
        filings_2020["cik"] = pd.to_numeric(
            filings_2020["cik"], errors="coerce"
        ).astype("Int64")
        filings_2020["word_count"] = pd.to_numeric(
            filings_2020["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2020)

        candidates_2020 = filings_2020.loc[
            filings_2020["word_count"] > 50_000
        ].reset_index(drop=True)
        tracker.record("filter", len(filings_2020), candidates_2020)

        documents_2020 = load_selected_texts(
            DATASET,
            candidates_2020,
            path_column="text",
            output_column="document_2020",
        )[["cik", "document_2020"]]
        tracker.record(
            "scan",
            len(candidates_2020),
            _intermediate_view(documents_2020),
        )

        software_plan = memory_dataset(
            f"{TASK_ID}-software", documents_2020
        ).sem_filter(
            filter=(
                "Keep the filing only if the reporting company's primary business "
                "is software or operating a technology platform, rather than merely "
                "using software or technology in another type of business."
            ),
            depends_on=["document_2020"],
        )
        started = time.time()
        software_result = software_plan.run(config)
        software_filings = result_frame(software_result, documents_2020)
        tracker.record_semantic(
            "sem_filter",
            len(documents_2020),
            _intermediate_view(software_filings),
            software_result,
            time.time() - started,
        )

        event_plan = memory_dataset(
            f"{TASK_ID}-acquisitions", software_filings
        ).sem_map(
            cols=[
                {
                    "name": "acquirer",
                    "type": str,
                    "desc": "The reporting acquirer's legal registrant name.",
                },
                {
                    "name": "acquirer_sector",
                    "type": str,
                    "desc": "The reporting acquirer's concise industry sector.",
                },
                {
                    "name": "acquired_company_names",
                    "type": list[str],
                    "desc": (
                        "Distinct names of whole companies the reporting company "
                        "acquired or entered a definitive agreement to acquire. "
                        "Exclude asset purchases, minority-stake investments, "
                        "spin-offs, and SPAC or blank-check combinations; use an "
                        "empty list when none qualify."
                    ),
                },
            ],
            desc=(
                "Extract the acquirer legal name, its industry sector, and names of "
                "qualifying whole-company acquisitions."
            ),
            depends_on=["document_2020"],
        )
        started = time.time()
        event_result = event_plan.run(config)
        extracted_events = result_frame(
            event_result,
            software_filings,
            ["acquirer", "acquirer_sector", "acquired_company_names"],
        )
        extracted_events["acquired_company_names"] = extracted_events[
            "acquired_company_names"
        ].map(_string_items)
        tracker.record_semantic(
            "sem_map",
            len(software_filings),
            _intermediate_view(extracted_events),
            event_result,
            time.time() - started,
        )

        acquisition_events = extracted_events.loc[
            extracted_events["acquired_company_names"].map(len).gt(0)
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(extracted_events),
            _intermediate_view(acquisition_events),
        )

        filings_2019 = load_table(
            DATASET,
            "CSV/2019.csv",
            ["cik", "text", "word_count"],
        )
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        filings_2019["word_count"] = pd.to_numeric(
            filings_2019["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, filings_2019)

        candidates_2019 = filings_2019.loc[
            filings_2019["word_count"] > 50_000
        ].reset_index(drop=True)
        tracker.record("filter", len(filings_2019), candidates_2019)

        documents_2019 = load_selected_texts(
            DATASET,
            candidates_2019,
            path_column="text",
            output_column="document_2019",
        )[["cik", "document_2019"]]
        tracker.record(
            "scan",
            len(candidates_2019),
            _intermediate_view(documents_2019),
        )

        registry_plan = memory_dataset(
            f"{TASK_ID}-registry-2019", documents_2019
        ).sem_map(
            cols=[
                {
                    "name": "registry_company_name",
                    "type": str,
                    "desc": (
                        "The legal registrant name, or null when it cannot be "
                        "determined."
                    ),
                },
                {
                    "name": "target_sector",
                    "type": str,
                    "desc": (
                        "The registrant's concise industry sector, or null when it "
                        "cannot be determined."
                    ),
                },
                {
                    "name": "target_risk_factor_count",
                    "type": int,
                    "desc": (
                        "The total number of distinct risk factors in the filing's "
                        "Risk Factors section, or null when it cannot be determined."
                    ),
                },
            ],
            desc=(
                "Extract the registrant legal name, industry sector, and total "
                "number of risk factors."
            ),
            depends_on=["document_2019"],
        )
        started = time.time()
        registry_result = registry_plan.run(config)
        registry_2019 = result_frame(
            registry_result,
            documents_2019,
            [
                "registry_company_name",
                "target_sector",
                "target_risk_factor_count",
            ],
        )
        registry_2019["registry_company_name"] = registry_2019[
            "registry_company_name"
        ].map(_clean_nullable_text)
        registry_2019["target_sector"] = registry_2019["target_sector"].map(
            _clean_nullable_text
        )
        registry_2019["target_risk_factor_count"] = pd.to_numeric(
            registry_2019["target_risk_factor_count"], errors="coerce"
        ).astype("Int64")
        tracker.record_semantic(
            "sem_map",
            len(documents_2019),
            _intermediate_view(registry_2019),
            registry_result,
            time.time() - started,
        )

        left_bindings = acquisition_events[
            [
                "cik",
                "acquirer",
                "acquirer_sector",
                "acquired_company_names",
                "document_2020",
            ]
        ].rename(columns={"cik": "acquirer_cik"})
        right_bindings = registry_2019[
            [
                "cik",
                "registry_company_name",
                "target_sector",
                "target_risk_factor_count",
                "document_2019",
            ]
        ].rename(columns={"cik": "registry_cik"})

        join_plan = memory_dataset(
            f"{TASK_ID}-acquisition-events", left_bindings
        ).sem_join(
            memory_dataset(f"{TASK_ID}-registry", right_bindings),
            condition=(
                "Match only when at least one acquired-company name identifies the "
                "same legal entity as the registry company. Allow abbreviations, "
                "former names, spelling variants, and legal-suffix differences, "
                "but reject merely similar unrelated names; use filing context only "
                "to resolve identity."
            ),
            depends_on=[
                "acquired_company_names",
                "document_2020",
                "registry_company_name",
                "document_2019",
            ],
        )
        started = time.time()
        join_result = join_plan.run(config)
        matched = result_frame(join_result)
        if matched.empty:
            matched = pd.DataFrame(columns=JOIN_COLUMNS)
        tracker.record_semantic(
            "sem_join",
            {"left": len(left_bindings), "right": len(right_bindings)},
            _intermediate_view(matched),
            join_result,
            time.time() - started,
        )

        acquirer_sector = (
            matched["acquirer_sector"].fillna("").astype(str).str.strip().str.casefold()
        )
        target_sector = (
            matched["target_sector"].fillna("").astype(str).str.strip().str.casefold()
        )
        qualifying = matched.loc[
            matched["acquirer_cik"].notna()
            & matched["registry_cik"].notna()
            & matched["acquirer_cik"].ne(matched["registry_cik"])
            & acquirer_sector.ne("")
            & target_sector.ne("")
            & acquirer_sector.ne(target_sector)
            & matched["target_risk_factor_count"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "filter",
            len(matched),
            _intermediate_view(qualifying),
        )

        projected = qualifying[
            [
                "acquirer",
                "registry_company_name",
                "target_sector",
                "target_risk_factor_count",
            ]
        ].rename(columns={"registry_company_name": "target"})
        tracker.record("project", len(qualifying), projected)

        answer_frame = (
            projected.assign(
                _acquirer_key=projected["acquirer"].astype(str).str.casefold(),
                _target_key=projected["target"].astype(str).str.casefold(),
            )
            .drop_duplicates(["_acquirer_key", "_target_key"], keep="first")
            .drop(columns=["_acquirer_key", "_target_key"])
            .reset_index(drop=True)
        )
        tracker.record("dedup", len(projected), answer_frame)
        answer = df_records(answer_frame)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
