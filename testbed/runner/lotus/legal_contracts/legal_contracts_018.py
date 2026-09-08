#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-018."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_docs,
    normalize_enum,
    save_output,
    setup,
)


TASK_ID = "legal_contracts-018"
AGREEMENT_TYPES = (
    "sox_certification",
    "other",
    "employment_agreement",
    "insider_trading_policy",
    "auditor_consent",
    "subsidiary_list",
    "clawback_policy",
    "mutual_nda",
    "confidentiality_standstill",
    "unilateral_nda",
    "non_compete_agreement",
    "supply_agreement",
    "non_solicitation_agreement",
)


def main():
    setup(max_tokens=256)
    tracker = StepTracker()

    with Timer() as timer:
        documents = load_docs("contract-exhibit").rename(
            columns={"doc_id": "contract_id", "contents": "text"}
        )
        documents["doc_format"] = documents["contract_id"].map(
            lambda value: os.path.splitext(str(value))[1].lstrip(".").lower()
        )
        documents = documents[["contract_id", "doc_format", "text"]]
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all)",
            None,
            len(documents),
            output=documents,
        )

        html_documents = documents[documents["doc_format"] == "htm"].copy()
        tracker.record(
            "FILTER(doc_format='htm')",
            len(documents),
            len(html_documents),
            output=html_documents,
        )

        with tracker.step(
            "SEM_CLASSIFY(primary agreement or filing type)",
            input_rows=len(html_documents),
        ) as step:
            classified = html_documents.sem_map(
                "Assign exactly one primary agreement_or_filing_type to {text}. "
                "Output exactly one of sox_certification, other, "
                "employment_agreement, insider_trading_policy, auditor_consent, "
                "subsidiary_list, clawback_policy, mutual_nda, "
                "confidentiality_standstill, unilateral_nda, "
                "non_compete_agreement, supply_agreement, or "
                "non_solicitation_agreement. Treat genuine Section 302 or "
                "Section 906 officer certifications as sox_certification, but "
                "classify Regulation AB servicing certifications as other. "
                "Output only the label.",
                suffix="agreement_or_filing_type",
            )
            classified["agreement_or_filing_type"] = classified[
                "agreement_or_filing_type"
            ].map(lambda value: normalize_enum(value, AGREEMENT_TYPES) or "other")
            step.set_output(classified)

        grouped = (
            classified.groupby("agreement_or_filing_type", sort=False)
            .agg(document_count=("contract_id", "nunique"))
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(agreement_or_filing_type, COUNT_DISTINCT(contract_id))",
            len(classified),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["document_count", "agreement_or_filing_type"],
            ascending=[False, True],
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(document_count DESC, agreement_or_filing_type ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
