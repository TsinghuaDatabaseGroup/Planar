#!/usr/bin/env python3
"""Palimpzest pipeline for finance-041."""

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

TASK_ID = "finance-041"
DATASET = "SEC"
AVAILABLE_YEARS = tuple(range(2019, 2025))
JOIN_COLUMNS = [
    "acquirer",
    "acquirer_sector",
    "acquired_company_names",
    "registry_company_name",
    "target_sector",
]


def _intermediate_view(frame):
    return frame.drop(
        columns=["document_text", "registry_document_text"],
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
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            items.append(item)
    return items


def main() -> None:
    config = get_config(max_tokens=4096)
    tracker = StepTracker(TASK_ID)

    with Timer() as timer:
        acquisition_filings = pd.concat(
            [
                load_table(DATASET, f"CSV/{year}.csv", ["text"])
                for year in AVAILABLE_YEARS
            ],
            ignore_index=True,
        )
        tracker.record("scan", None, acquisition_filings)

        acquisition_documents = load_selected_texts(
            DATASET,
            acquisition_filings,
            path_column="text",
            output_column="document_text",
        )[["document_text"]]
        tracker.record(
            "scan",
            len(acquisition_filings),
            _intermediate_view(acquisition_documents),
        )

        software_plan = memory_dataset(
            f"{TASK_ID}-software", acquisition_documents
        ).sem_filter(
            filter=(
                "Keep the filing only if the reporting company's primary business "
                "is software or operating a technology platform. Do not qualify a "
                "company merely because it uses software or technology in another "
                "type of business."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        software_result = software_plan.run(config)
        software_filings = result_frame(software_result, acquisition_documents)
        tracker.record_semantic(
            "sem_filter",
            len(acquisition_documents),
            _intermediate_view(software_filings),
            software_result,
            time.time() - started,
        )

        event_plan = memory_dataset(
            f"{TASK_ID}-acquisition-events", software_filings
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
                    "desc": "The reporting acquirer's concise primary industry sector.",
                },
                {
                    "name": "acquired_company_names",
                    "type": list[str],
                    "desc": (
                        "Distinct legal or commonly stated names of whole companies "
                        "the reporting company acquired or entered a definitive "
                        "agreement to acquire. Exclude asset purchases, minority-"
                        "stake investments, spin-offs, and SPAC or blank-check "
                        "business combinations; use an empty list when none qualify."
                    ),
                },
            ],
            desc=(
                "Extract the acquirer legal name, its primary industry sector, and "
                "qualifying whole-company acquisition target names."
            ),
            depends_on=["document_text"],
        )
        started = time.time()
        event_result = event_plan.run(config)
        extracted_events = result_frame(
            event_result,
            software_filings,
            ["acquirer", "acquirer_sector", "acquired_company_names"],
        )
        extracted_events["acquirer"] = extracted_events["acquirer"].map(
            _clean_text
        )
        extracted_events["acquirer_sector"] = extracted_events[
            "acquirer_sector"
        ].map(_clean_text)
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

        registry_filings = pd.concat(
            [
                load_table(
                    DATASET,
                    f"CSV/{year}.csv",
                    ["cik", "year", "text", "word_count"],
                )
                for year in AVAILABLE_YEARS
            ],
            ignore_index=True,
        )
        registry_filings["cik"] = pd.to_numeric(
            registry_filings["cik"], errors="coerce"
        ).astype("Int64")
        registry_filings["year"] = pd.to_numeric(
            registry_filings["year"], errors="coerce"
        ).astype("Int64")
        registry_filings["word_count"] = pd.to_numeric(
            registry_filings["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record("scan", None, registry_filings)

        registry_documents = (
            registry_filings.sort_values(
                ["cik", "year", "word_count"],
                ascending=[True, True, False],
                kind="mergesort",
            )
            .drop_duplicates("cik", keep="first")
            .rename(columns={"text": "registry_text"})[
                ["cik", "year", "word_count", "registry_text"]
            ]
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(registry_filings), registry_documents)

        registry_document_texts = load_selected_texts(
            DATASET,
            registry_documents,
            path_column="registry_text",
            output_column="registry_document_text",
        )[["cik", "year", "word_count", "registry_document_text"]]
        tracker.record(
            "scan",
            len(registry_documents),
            _intermediate_view(registry_document_texts),
        )

        registry_plan = memory_dataset(
            f"{TASK_ID}-company-registry", registry_document_texts
        ).sem_map(
            cols=[
                {
                    "name": "registry_company_name",
                    "type": str,
                    "desc": "The filing company's legal registrant name.",
                },
                {
                    "name": "target_sector",
                    "type": str,
                    "desc": "The filing company's concise primary industry sector.",
                },
            ],
            desc=(
                "Extract the filing company's legal registrant name and primary "
                "industry sector."
            ),
            depends_on=["registry_document_text"],
        )
        started = time.time()
        registry_result = registry_plan.run(config)
        company_registry = result_frame(
            registry_result,
            registry_document_texts,
            ["registry_company_name", "target_sector"],
        )
        company_registry["registry_company_name"] = company_registry[
            "registry_company_name"
        ].map(_clean_text)
        company_registry["target_sector"] = company_registry[
            "target_sector"
        ].map(_clean_text)
        tracker.record_semantic(
            "sem_map",
            len(registry_document_texts),
            _intermediate_view(company_registry),
            registry_result,
            time.time() - started,
        )

        left_bindings = acquisition_events[
            ["acquirer", "acquirer_sector", "acquired_company_names"]
        ].copy()
        right_bindings = company_registry[
            ["registry_company_name", "target_sector"]
        ].copy()

        join_plan = memory_dataset(
            f"{TASK_ID}-events", left_bindings
        ).sem_join(
            memory_dataset(f"{TASK_ID}-registry", right_bindings),
            condition=(
                "Match only when at least one acquired-company name identifies the "
                "same legal entity as the registry company. Allow harmless naming "
                "variations such as abbreviations, former names, spelling variants, "
                "and legal-suffix differences, but reject merely similar unrelated "
                "names."
            ),
            depends_on=["acquired_company_names", "registry_company_name"],
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

        acquirer_names = matched["acquirer"].fillna("").astype(str).str.casefold()
        target_names = (
            matched["registry_company_name"].fillna("").astype(str).str.casefold()
        )
        acquirer_sectors = (
            matched["acquirer_sector"].fillna("").astype(str).str.casefold()
        )
        target_sectors = (
            matched["target_sector"].fillna("").astype(str).str.casefold()
        )
        qualifying = matched.loc[
            acquirer_names.ne("")
            & target_names.ne("")
            & acquirer_names.ne(target_names)
            & acquirer_sectors.ne("")
            & target_sectors.ne("")
            & acquirer_sectors.ne(target_sectors)
        ].reset_index(drop=True)
        tracker.record("filter", len(matched), qualifying)

        projected = qualifying[
            ["acquirer", "registry_company_name", "target_sector"]
        ].rename(columns={"registry_company_name": "target"})
        tracker.record("project", len(qualifying), projected)

        answer_frame = (
            projected.assign(
                _acquirer_key=projected["acquirer"].str.casefold(),
                _target_key=projected["target"].str.casefold(),
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
