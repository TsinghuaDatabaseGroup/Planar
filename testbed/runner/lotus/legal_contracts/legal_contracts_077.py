#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-077."""

import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_number,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-077"
LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "llp",
    "lp",
    "ltd",
    "plc",
}


def _normalize_company_name(value):
    tokens = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _build_party_index(records, id_column, party_column):
    index = defaultdict(list)
    for record in records:
        seen = set()
        for stated_name in record[party_column]:
            key = _normalize_company_name(stated_name)
            identity = (key, stated_name)
            if not key or identity in seen:
                continue
            seen.add(identity)
            index[key].append((record, stated_name))
    return index


def _join_records(nda_records, restrictive_records):
    restrictive_index = _build_party_index(
        restrictive_records,
        "restrictive_document_id",
        "restrictive_company_parties",
    )
    matches = {}
    for nda in nda_records:
        for nda_name in nda["nda_company_parties"]:
            key = _normalize_company_name(nda_name)
            if not key:
                continue
            for restrictive, restrictive_name in restrictive_index.get(key, []):
                if nda["nda_document_id"] == restrictive["restrictive_document_id"]:
                    continue
                pair_key = (
                    nda["nda_document_id"],
                    restrictive["restrictive_document_id"],
                )
                match = matches.setdefault(
                    pair_key,
                    {
                        "matched_stated_variants": set(),
                        "nda_document_id": nda["nda_document_id"],
                        "restrictive_document_id": restrictive[
                            "restrictive_document_id"
                        ],
                        "nda_duration_years": nda["nda_duration_years"],
                        "noncompete_duration_years": restrictive[
                            "noncompete_duration_years"
                        ],
                    },
                )
                match["matched_stated_variants"].update(
                    (nda_name, restrictive_name)
                )

    output = []
    for match in matches.values():
        match = dict(match)
        match["matched_stated_variants"] = sorted(
            match["matched_stated_variants"]
        )
        output.append(match)
    return output


def main():
    setup(max_tokens=1024, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        nda_docs = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS nda_docs",
            None,
            len(nda_docs),
            output=nda_docs,
        )

        with tracker.step(
            "SEM_FILTER(NDA with finite confidentiality term)",
            input_rows=len(nda_docs),
        ) as step:
            nda_agreements = nda_docs.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement with a finite stated confidentiality term."
            ).reset_index(drop=True)
            step.set_output(nda_agreements)

        with tracker.step(
            "SEM_EXTRACT(NDA parties and confidentiality duration)",
            input_rows=len(nda_agreements),
        ) as step:
            nda_extracted = nda_agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "nda_company_parties": (
                        "a list of all corporate party names stated in the agreement"
                    ),
                    "nda_duration_years": (
                        "the finite confidentiality duration normalized to years as "
                        "a number"
                    ),
                },
            )
            nda_extracted["nda_company_parties"] = nda_extracted[
                "nda_company_parties"
            ].map(parse_string_list)
            nda_extracted["nda_duration_years"] = nda_extracted[
                "nda_duration_years"
            ].map(parse_number)
            nda_extracted = nda_extracted.rename(
                columns={"document_id": "nda_document_id"}
            )[
                [
                    "nda_document_id",
                    "nda_company_parties",
                    "nda_duration_years",
                ]
            ].reset_index(drop=True)
            step.set_output(nda_extracted)

        restrictive_docs = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS restrictive_docs",
            None,
            len(restrictive_docs),
            output=restrictive_docs,
        )

        with tracker.step(
            "SEM_FILTER(employment or standalone restrictive-covenant agreement)",
            input_rows=len(restrictive_docs),
        ) as step:
            restrictive_agreements = restrictive_docs.sem_filter(
                "The document {text} is an employment agreement or a standalone "
                "restrictive-covenant agreement."
            ).reset_index(drop=True)
            step.set_output(restrictive_agreements)

        with tracker.step(
            "SEM_FILTER(operative non-compete with finite duration)",
            input_rows=len(restrictive_agreements),
        ) as step:
            finite_noncompetes = restrictive_agreements.sem_filter(
                "The agreement {text} contains an operative non-compete clause with "
                "a finite stated duration."
            ).reset_index(drop=True)
            step.set_output(finite_noncompetes)

        with tracker.step(
            "SEM_EXTRACT(restrictive-agreement parties and non-compete duration)",
            input_rows=len(finite_noncompetes),
        ) as step:
            restrictive_extracted = finite_noncompetes.sem_extract(
                input_cols=["text"],
                output_cols={
                    "restrictive_company_parties": (
                        "a list of all corporate party names stated in the agreement"
                    ),
                    "noncompete_duration_years": (
                        "the finite non-compete duration normalized to years as a number"
                    ),
                },
            )
            restrictive_extracted["restrictive_company_parties"] = (
                restrictive_extracted["restrictive_company_parties"].map(
                    parse_string_list
                )
            )
            restrictive_extracted["noncompete_duration_years"] = (
                restrictive_extracted["noncompete_duration_years"].map(parse_number)
            )
            restrictive_extracted = restrictive_extracted.rename(
                columns={"document_id": "restrictive_document_id"}
            )[
                [
                    "restrictive_document_id",
                    "restrictive_company_parties",
                    "noncompete_duration_years",
                ]
            ].reset_index(drop=True)
            step.set_output(restrictive_extracted)

        nda_records = df_records(nda_extracted)
        restrictive_records = df_records(restrictive_extracted)
        joined = _join_records(nda_records, restrictive_records)
        tracker.record(
            "JOIN(nda_records, restrictive_records, on=normalized company intersection and distinct document IDs)",
            {
                "nda_records": len(nda_records),
                "restrictive_records": len(restrictive_records),
            },
            len(joined),
            output=joined,
        )

        projected = [
            {
                "company_name": record["matched_stated_variants"][0],
                "nda_document_id": record["nda_document_id"],
                "restrictive_document_id": record["restrictive_document_id"],
                "nda_duration_years": record["nda_duration_years"],
                "noncompete_duration_years": record[
                    "noncompete_duration_years"
                ],
            }
            for record in joined
        ]
        tracker.record(
            "PROJECT(smallest matched company variant and both document durations)",
            len(joined),
            len(projected),
            output=projected,
        )
        answer = projected

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
