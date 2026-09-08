#!/usr/bin/env python3
"""LOTUS pipeline for finance-040."""

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

TASK_ID = "finance-040"
BATCH_SIZE = 100
YEARS = tuple(range(2019, 2025))


def _empty_documents(records: pd.DataFrame) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty["document_text"] = pd.Series(dtype="object")
    return empty


def _without_large_bindings(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            "document_text",
            "counterparty_join_binding",
            "registry_join_binding",
        ],
        errors="ignore",
    )


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
        tracker.record("SCAN_TABLE(CSV)", None, len(filings), output=filings)
        tracker.record(
            "SCAN_DOCS(selector=filings.text)",
            len(filings),
            len(filings),
            output=filings,
        )

        with tracker.step(
            "SEM_FILTER(Software industry sector)",
            input_rows=len(filings),
        ) as step:
            software_parts = []
            for documents in iter_selected_texts(
                "SEC",
                filings,
                path_column="text",
                output_column="document_text",
                batch_size=BATCH_SIZE,
            ):
                software_parts.append(
                    documents.sem_filter(
                        "The reporting company in the 10-K filing {document_text} "
                        "belongs to the Software industry sector."
                    )
                )
            software_filings = (
                pd.concat(software_parts, ignore_index=True)
                if software_parts
                else _empty_documents(filings)
            )
            step.set_output(_without_large_bindings(software_filings))

        with tracker.step(
            "SEM_EXTRACT(software company and whole-company M&A counterparties)",
            input_rows=len(software_filings),
        ) as step:
            software_events = sem_extract_in_batches(
                software_filings,
                input_cols=["document_text"],
                output_cols={
                    "company_name": (
                        "the reporting company's legal name stated in the filing"
                    ),
                    "counterparty_names": (
                        "a JSON list of distinct legal or commonly stated names of "
                        "whole-company merger or acquisition counterparties disclosed "
                        "in the filing; exclude asset purchases, minority-stake "
                        "investments, spin-offs, single-company going-private "
                        "recapitalizations, and SPAC or blank-check combinations; use "
                        "an empty list when none qualify"
                    ),
                },
                batch_size=BATCH_SIZE,
            )
            software_events["company_name"] = software_events["company_name"].map(
                clean_text
            )
            software_events["counterparty_names"] = software_events[
                "counterparty_names"
            ].map(parse_string_list)
            step.set_output(_without_large_bindings(software_events))

        candidate_events = software_events[
            software_events["counterparty_names"].map(len) > 0
        ].copy()
        tracker.record(
            "FILTER(LENGTH(counterparty_names) > 0)",
            len(software_events),
            len(candidate_events),
            output=_without_large_bindings(candidate_events),
        )

        software_registry = (
            software_events[["company_name"]]
            .assign(
                _company_key=software_events["company_name"].str.casefold()
            )
            .drop_duplicates("_company_key", keep="first")
            .drop(columns=["_company_key"])
            .rename(columns={"company_name": "registry_company_name"})
        )
        tracker.record(
            "DEDUP(software registry company_name)",
            len(software_events),
            len(software_registry),
            output=software_registry,
        )

        left_bindings = candidate_events[["company_name", "counterparty_names"]].copy()
        left_bindings["counterparty_join_binding"] = left_bindings[
            "counterparty_names"
        ].map(lambda names: json.dumps(names, ensure_ascii=False))
        right_bindings = software_registry.copy()
        right_bindings["registry_join_binding"] = right_bindings[
            "registry_company_name"
        ].astype(str)
        tracker.record(
            "CODE_MAP(bind counterparty SEM_JOIN inputs)",
            len(candidate_events),
            len(left_bindings),
            output=_without_large_bindings(left_bindings),
        )
        tracker.record(
            "CODE_MAP(bind software-registry SEM_JOIN inputs)",
            len(software_registry),
            len(right_bindings),
            output=_without_large_bindings(right_bindings),
        )

        with tracker.step(
            "SEM_JOIN(counterparty to same software legal entity)",
            input_rows={"left": len(left_bindings), "right": len(right_bindings)},
        ) as step:
            if left_bindings.empty or right_bindings.empty:
                matched = pd.DataFrame(
                    columns=[*left_bindings.columns, *right_bindings.columns]
                )
            else:
                matched = left_bindings.sem_join(
                    right_bindings,
                    "Match at least one named whole-company M&A counterparty in "
                    "{counterparty_join_binding} to {registry_join_binding} only when "
                    "they identify the same legal entity despite harmless "
                    "abbreviations, spelling variants, former names, or legal-suffix "
                    "differences."
                )
            step.set_output(_without_large_bindings(matched))

        distinct_matches = matched[
            matched["company_name"].str.casefold()
            != matched["registry_company_name"].str.casefold()
        ].copy()
        tracker.record(
            "FILTER(candidate company_name != registry company_name)",
            len(matched),
            len(distinct_matches),
            output=_without_large_bindings(distinct_matches),
        )

        pair_rows = []
        for company_name, registry_name in zip(
            distinct_matches["company_name"],
            distinct_matches["registry_company_name"],
        ):
            if str(company_name).casefold() <= str(registry_name).casefold():
                company_a, company_b = company_name, registry_name
            else:
                company_a, company_b = registry_name, company_name
            pair_rows.append({"company_a": company_a, "company_b": company_b})
        projected = pd.DataFrame(pair_rows, columns=["company_a", "company_b"])
        tracker.record(
            "PROJECT(case-insensitive alphabetized company_a, company_b)",
            len(distinct_matches),
            len(projected),
            output=projected,
        )

        answer_frame = (
            projected.assign(
                _company_a_key=projected["company_a"].str.casefold(),
                _company_b_key=projected["company_b"].str.casefold(),
            )
            .drop_duplicates(["_company_a_key", "_company_b_key"], keep="first")
            .drop(columns=["_company_a_key", "_company_b_key"])
            .reset_index(drop=True)
        )
        tracker.record(
            "DEDUP(company_a, company_b)",
            len(projected),
            len(answer_frame),
            output=answer_frame,
        )
        answer = df_records(answer_frame)

    print(f"Result: {len(answer)} pairs")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
