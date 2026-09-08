#!/usr/bin/env python3
"""LOTUS pipeline for finance-041."""

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

TASK_ID = "finance-041"
BATCH_SIZE = 100
YEARS = tuple(range(2019, 2025))


def _load_all_filings(columns: list[str]) -> pd.DataFrame:
    parts = []
    for year in YEARS:
        part = load_table("SEC", f"CSV/{year}.csv")[columns].copy()
        if "year" in columns:
            part["year"] = year
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _empty_documents(records: pd.DataFrame, column: str) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty[column] = pd.Series(dtype="object")
    return empty


def _without_large_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            "document_text",
            "registry_document_text",
            "acquisition_join_binding",
            "registry_join_binding",
        ],
        errors="ignore",
    )


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        acquisition_filings = _load_all_filings(["text"])
        tracker.record(
            "SCAN_TABLE(CSV AS acquisition_filings)",
            None,
            len(acquisition_filings),
            output=acquisition_filings,
        )
        tracker.record(
            "SCAN_DOCS(selector=acquisition_filings.text)",
            len(acquisition_filings),
            len(acquisition_filings),
            output=acquisition_filings,
        )

        with tracker.step(
            "SEM_FILTER(software or technology-platform primary business)",
            input_rows=len(acquisition_filings),
        ) as step:
            software_parts = []
            for documents in iter_selected_texts(
                "SEC",
                acquisition_filings,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                software_parts.append(
                    documents.sem_filter(
                        "The reporting company's primary business in the 10-K "
                        "filing {document_text} is software or operating a technology "
                        "platform, rather than merely using software or technology "
                        "in another type of business."
                    )
                )
            software_filings = (
                pd.concat(software_parts, ignore_index=True)
                if software_parts
                else _empty_documents(acquisition_filings, "document_text")
            )
            step.set_output(_without_large_columns(software_filings))

        with tracker.step(
            "SEM_EXTRACT(acquirer, sector, and whole-company acquisitions)",
            input_rows=len(software_filings),
        ) as step:
            extracted_events = sem_extract_in_batches(
                software_filings,
                input_cols=["document_text"],
                output_cols={
                    "acquirer": (
                        "the reporting acquirer's legal registrant name stated in "
                        "the filing"
                    ),
                    "acquirer_sector": (
                        "the reporting acquirer's concise primary industry sector"
                    ),
                    "acquired_company_names": (
                        "a JSON list of distinct legal or commonly stated names of "
                        "whole companies that the reporting company acquired or "
                        "entered a definitive agreement to acquire; exclude asset "
                        "purchases, minority-stake investments, spin-offs, and SPAC "
                        "or blank-check business combinations; use an empty list "
                        "when none qualify"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            extracted_events["acquirer"] = extracted_events["acquirer"].map(
                clean_text
            )
            extracted_events["acquirer_sector"] = extracted_events[
                "acquirer_sector"
            ].map(clean_text)
            extracted_events["acquired_company_names"] = extracted_events[
                "acquired_company_names"
            ].map(parse_string_list)
            step.set_output(_without_large_columns(extracted_events))

        acquisition_events = extracted_events[
            extracted_events["acquired_company_names"].map(len) > 0
        ].copy()
        tracker.record(
            "FILTER(LENGTH(acquired_company_names) > 0)",
            len(extracted_events),
            len(acquisition_events),
            output=_without_large_columns(acquisition_events),
        )

        registry_filings = _load_all_filings(
            ["cik", "year", "text", "word_count"]
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
        tracker.record(
            "SCAN_TABLE(CSV AS registry_filings)",
            None,
            len(registry_filings),
            output=registry_filings,
        )

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
        tracker.record(
            "GROUP_BY(cik, MIN_BY(text, year ASC, word_count DESC) AS registry_text)",
            len(registry_filings),
            len(registry_documents),
            output=registry_documents,
        )
        tracker.record(
            "SCAN_DOCS(selector=registry_text)",
            len(registry_documents),
            len(registry_documents),
            output=registry_documents,
        )

        with tracker.step(
            "SEM_EXTRACT(registry company legal name and target sector)",
            input_rows=len(registry_documents),
        ) as step:
            registry_parts = []
            for documents in iter_selected_texts(
                "SEC",
                registry_documents,
                path_column="registry_text",
                output_column="registry_document_text",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["registry_document_text"],
                    output_cols={
                        "registry_company_name": (
                            "the filing company's legal registrant name"
                        ),
                        "target_sector": (
                            "the filing company's concise primary industry sector"
                        ),
                    },
                )
                batch["registry_company_name"] = batch[
                    "registry_company_name"
                ].map(clean_text)
                batch["target_sector"] = batch["target_sector"].map(clean_text)
                registry_parts.append(batch)
            company_registry = (
                pd.concat(registry_parts, ignore_index=True)
                if registry_parts
                else _empty_documents(
                    registry_documents, "registry_document_text"
                ).assign(
                    registry_company_name=pd.Series(dtype="object"),
                    target_sector=pd.Series(dtype="object"),
                )
            )
            step.set_output(_without_large_columns(company_registry))

        left_bindings = acquisition_events[
            ["acquirer", "acquirer_sector", "acquired_company_names"]
        ].copy()
        left_bindings["acquisition_join_binding"] = left_bindings[
            "acquired_company_names"
        ].map(lambda names: json.dumps(names, ensure_ascii=False))
        tracker.record(
            "CODE_MAP(bind acquired-company SEM_JOIN inputs)",
            len(acquisition_events),
            len(left_bindings),
            output=_without_large_columns(left_bindings),
        )

        right_bindings = company_registry[
            ["registry_company_name", "target_sector"]
        ].copy()
        right_bindings["registry_join_binding"] = right_bindings[
            "registry_company_name"
        ].astype(str)
        tracker.record(
            "CODE_MAP(bind company-registry SEM_JOIN inputs)",
            len(company_registry),
            len(right_bindings),
            output=_without_large_columns(right_bindings),
        )

        with tracker.step(
            "SEM_JOIN(acquired-company name to same registry legal entity)",
            input_rows={"left": len(left_bindings), "right": len(right_bindings)},
        ) as step:
            if left_bindings.empty or right_bindings.empty:
                matched = pd.DataFrame(
                    columns=[*left_bindings.columns, *right_bindings.columns]
                )
            else:
                matched = left_bindings.sem_join(
                    right_bindings,
                    "Match at least one acquired-company name in "
                    "{acquisition_join_binding} to {registry_join_binding} only "
                    "when they identify the same legal entity. Allow harmless "
                    "abbreviations, spelling variants, former names, and "
                    "legal-suffix differences, but reject merely similar unrelated "
                    "names."
                )
            step.set_output(_without_large_columns(matched))

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
        qualifying = matched[
            acquirer_names.ne("")
            & target_names.ne("")
            & acquirer_names.ne(target_names)
            & acquirer_sectors.ne("")
            & target_sectors.ne("")
            & acquirer_sectors.ne(target_sectors)
        ].copy()
        tracker.record(
            "FILTER(acquirer != registry company AND acquirer sector != target sector)",
            len(matched),
            len(qualifying),
            output=_without_large_columns(qualifying),
        )

        projected = qualifying[
            ["acquirer", "registry_company_name", "target_sector"]
        ].rename(columns={"registry_company_name": "target"})
        tracker.record(
            "PROJECT(acquirer, target, target_sector)",
            len(qualifying),
            len(projected),
            output=projected,
        )

        answer_frame = (
            projected.assign(
                _acquirer_key=projected["acquirer"].str.casefold(),
                _target_key=projected["target"].str.casefold(),
            )
            .drop_duplicates(["_acquirer_key", "_target_key"], keep="first")
            .drop(columns=["_acquirer_key", "_target_key"])
            .reset_index(drop=True)
        )
        tracker.record(
            "DEDUP(acquirer, target)",
            len(projected),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
