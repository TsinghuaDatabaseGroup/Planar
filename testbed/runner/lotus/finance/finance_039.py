#!/usr/bin/env python3
"""LOTUS pipeline for finance-039."""

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
    parse_number,
    parse_string_list,
    save_output,
    sem_extract_in_batches,
    setup,
)

TASK_ID = "finance-039"
BATCH_SIZE = 100


def _empty_documents(records: pd.DataFrame, column: str) -> pd.DataFrame:
    empty = records.iloc[0:0].copy()
    empty[column] = pd.Series(dtype="object")
    return empty


def _clean_nullable_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = clean_text(value, default="")
    return text or None


def _without_documents(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            "document_2020",
            "document_2019",
            "acquisition_join_binding",
            "registry_join_binding",
        ],
        errors="ignore",
    )


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings_2020 = load_table("SEC", "CSV/2020.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings_2020["cik"] = pd.to_numeric(
            filings_2020["cik"], errors="coerce"
        ).astype("Int64")
        filings_2020["word_count"] = pd.to_numeric(
            filings_2020["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2020.csv AS filings_2020)",
            None,
            len(filings_2020),
            output=filings_2020,
        )
        candidates_2020 = filings_2020[filings_2020["word_count"] > 50000].copy()
        tracker.record(
            "FILTER(2020 word_count > 50000)",
            len(filings_2020),
            len(candidates_2020),
            output=candidates_2020,
        )
        tracker.record(
            "SCAN_DOCS(selector=2020 filtered.text)",
            len(candidates_2020),
            len(candidates_2020),
            output=candidates_2020,
        )

        with tracker.step(
            "SEM_FILTER(software or technology-platform primary business)",
            input_rows=len(candidates_2020),
        ) as step:
            software_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates_2020,
                path_column="text",
                output_column="document_2020",
                batch_size=BATCH_SIZE,
            ):
                software_parts.append(
                    documents.sem_filter(
                        "The reporting company's primary business in the 2020 10-K "
                        "filing {document_2020} is software or operating a technology "
                        "platform, rather than merely using software or technology in "
                        "another type of business."
                    )
                )
            software_filings = (
                pd.concat(software_parts, ignore_index=True)
                if software_parts
                else _empty_documents(candidates_2020, "document_2020")
            )
            step.set_output(_without_documents(software_filings))

        with tracker.step(
            "SEM_EXTRACT(acquirer, sector, and whole-company acquisitions)",
            input_rows=len(software_filings),
        ) as step:
            extracted_events = sem_extract_in_batches(
                software_filings,
                input_cols=["document_2020"],
                output_cols={
                    "acquirer": (
                        "the reporting acquirer's legal registrant name stated in the "
                        "filing"
                    ),
                    "acquirer_sector": (
                        "the reporting acquirer's concise industry sector"
                    ),
                    "acquired_company_names": (
                        "a JSON list of distinct whole-company legal or commonly "
                        "stated names that the reporting company acquired or entered "
                        "a definitive agreement to acquire; exclude asset purchases, "
                        "minority-stake investments, spin-offs, and SPAC or blank-check "
                        "combinations; use an empty list when none qualify"
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
            step.set_output(_without_documents(extracted_events))

        acquisition_events = extracted_events[
            extracted_events["acquired_company_names"].map(len) > 0
        ].copy()
        tracker.record(
            "FILTER(LENGTH(acquired_company_names) > 0)",
            len(extracted_events),
            len(acquisition_events),
            output=_without_documents(acquisition_events),
        )

        filings_2019 = load_table("SEC", "CSV/2019.csv")[
            ["cik", "text", "word_count"]
        ].copy()
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        filings_2019["word_count"] = pd.to_numeric(
            filings_2019["word_count"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2019.csv AS filings_2019)",
            None,
            len(filings_2019),
            output=filings_2019,
        )
        candidates_2019 = filings_2019[filings_2019["word_count"] > 50000].copy()
        tracker.record(
            "FILTER(2019 word_count > 50000)",
            len(filings_2019),
            len(candidates_2019),
            output=candidates_2019,
        )
        tracker.record(
            "SCAN_DOCS(selector=2019 filtered.text)",
            len(candidates_2019),
            len(candidates_2019),
            output=candidates_2019,
        )

        with tracker.step(
            "SEM_EXTRACT(2019 registrant, sector, and risk-factor count)",
            input_rows=len(candidates_2019),
        ) as step:
            registry_parts = []
            for documents in iter_selected_texts(
                "SEC",
                candidates_2019,
                path_column="text",
                output_column="document_2019",
                batch_size=BATCH_SIZE,
            ):
                batch = documents.sem_extract(
                    input_cols=["document_2019"],
                    output_cols={
                        "registry_company_name": (
                            "the legal registrant name stated in the filing, or null "
                            "when it cannot be determined"
                        ),
                        "target_sector": (
                            "the registrant's concise industry sector, or null when it "
                            "cannot be determined"
                        ),
                        "target_risk_factor_count": (
                            "the total number of distinct risk factors in the filing's "
                            "Risk Factors section as an integer, or null when the total "
                            "cannot be determined"
                        ),
                    },
                )
                batch["registry_company_name"] = batch[
                    "registry_company_name"
                ].map(_clean_nullable_text)
                batch["target_sector"] = batch["target_sector"].map(
                    _clean_nullable_text
                )
                batch["target_risk_factor_count"] = pd.to_numeric(
                    batch["target_risk_factor_count"].map(parse_number),
                    errors="coerce",
                )
                registry_parts.append(batch)
            registry_2019 = (
                pd.concat(registry_parts, ignore_index=True)
                if registry_parts
                else _empty_documents(candidates_2019, "document_2019").assign(
                    registry_company_name=pd.Series(dtype="object"),
                    target_sector=pd.Series(dtype="object"),
                    target_risk_factor_count=pd.Series(dtype="float64"),
                )
            )
            step.set_output(_without_documents(registry_2019))

        left_bindings = acquisition_events[
            [
                "cik",
                "acquirer",
                "acquirer_sector",
                "acquired_company_names",
                "document_2020",
            ]
        ].rename(columns={"cik": "acquirer_cik"})
        left_bindings["acquisition_join_binding"] = [
            "acquirer="
            + str(acquirer)
            + "; acquired_company_names="
            + json.dumps(names, ensure_ascii=False)
            + "; acquisition_filing_context="
            + str(context)
            for acquirer, names, context in zip(
                left_bindings["acquirer"],
                left_bindings["acquired_company_names"],
                left_bindings["document_2020"],
            )
        ]
        left_bindings = left_bindings.drop(columns=["document_2020"])
        tracker.record(
            "CODE_MAP(bind acquisition SEM_JOIN inputs)",
            len(acquisition_events),
            len(left_bindings),
            output=_without_documents(left_bindings),
        )

        right_bindings = registry_2019[
            [
                "cik",
                "registry_company_name",
                "target_sector",
                "target_risk_factor_count",
                "document_2019",
            ]
        ].rename(columns={"cik": "registry_cik"})
        right_bindings["registry_join_binding"] = [
            "registry_company_name="
            + str(name)
            + "; registry_filing_context="
            + str(context)
            for name, context in zip(
                right_bindings["registry_company_name"],
                right_bindings["document_2019"],
            )
        ]
        right_bindings = right_bindings.drop(columns=["document_2019"])
        tracker.record(
            "CODE_MAP(bind registry SEM_JOIN inputs)",
            len(registry_2019),
            len(right_bindings),
            output=_without_documents(right_bindings),
        )

        with tracker.step(
            "SEM_JOIN(acquired-company reference to same 2019 legal entity)",
            input_rows={"left": len(left_bindings), "right": len(right_bindings)},
        ) as step:
            if left_bindings.empty or right_bindings.empty:
                matched = pd.DataFrame(
                    columns=[*left_bindings.columns, *right_bindings.columns]
                )
            else:
                matched = left_bindings.sem_join(
                    right_bindings,
                    "Match the 2020 acquisition record {acquisition_join_binding} "
                    "to the 2019 registrant {registry_join_binding} only when at "
                    "least one acquired-company reference identifies the same legal "
                    "entity as the registrant. Allow abbreviations, former names, "
                    "spelling variants, and legal-suffix differences, but reject "
                    "merely similar unrelated names; use filing context only to "
                    "resolve identity."
                )
            step.set_output(_without_documents(matched))

        acquirer_sector = matched["acquirer_sector"].fillna("").astype(str).str.strip().str.casefold()
        target_sector = matched["target_sector"].fillna("").astype(str).str.strip().str.casefold()
        qualifying = matched[
            matched["acquirer_cik"].notna()
            & matched["registry_cik"].notna()
            & (matched["acquirer_cik"] != matched["registry_cik"])
            & acquirer_sector.ne("")
            & target_sector.ne("")
            & acquirer_sector.ne(target_sector)
            & matched["target_risk_factor_count"].notna()
        ].copy()
        tracker.record(
            "FILTER(different CIK, different sector, target risk count IS NOT NULL)",
            len(matched),
            len(qualifying),
            output=_without_documents(qualifying),
        )

        projected = qualifying[
            [
                "acquirer",
                "registry_company_name",
                "target_sector",
                "target_risk_factor_count",
            ]
        ].rename(columns={"registry_company_name": "target"})
        tracker.record(
            "PROJECT(acquirer, target, target_sector, target_risk_factor_count)",
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
