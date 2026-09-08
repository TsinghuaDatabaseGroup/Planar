#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-019."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_docs,
    normalize_enum,
    parse_number,
    parse_record_list,
    save_output,
    setup,
)


TASK_ID = "legal_contracts-019"
AGREEMENT_TYPES = (
    "sox_certification",
    "employment_agreement",
    "insider_trading_policy",
    "auditor_consent",
    "subsidiary_list",
    "clawback_policy",
    "mutual_nda",
    "unilateral_nda",
    "confidentiality_standstill",
    "non_compete_agreement",
    "supply_agreement",
    "non_solicitation_agreement",
    "other",
)


def main():
    setup(max_tokens=1024)
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
            "SEM_AGGREGATE(top agreement or filing types)",
            input_rows=len(html_documents),
        ) as step:
            aggregated = html_documents[["contract_id", "text"]].sem_agg(
                "Across the complete input relation of contract IDs "
                "{contract_id} and documents {text}, assign every document "
                "exactly one primary agreement_or_filing_type from "
                "sox_certification, employment_agreement, "
                "insider_trading_policy, auditor_consent, subsidiary_list, "
                "clawback_policy, mutual_nda, unilateral_nda, "
                "confidentiality_standstill, non_compete_agreement, "
                "supply_agreement, non_solicitation_agreement, or other. Treat "
                "genuine Section 302 or Section 906 officer certifications as "
                "sox_certification, but classify Regulation AB servicing "
                "certifications as other. Count distinct documents per type and "
                "return the five most common types, ordered by document_count "
                "descending and then type alphabetically, with one-based rank. "
                "Output only a valid JSON array with keys rank, "
                "agreement_or_filing_type, and document_count, without markdown "
                "or commentary."
            )
            rows = []
            for item in parse_record_list(aggregated["_output"].iloc[0]):
                rank = parse_number(item.get("rank"))
                agreement_type = normalize_enum(
                    item.get("agreement_or_filing_type"), AGREEMENT_TYPES
                )
                count = parse_number(item.get("document_count"))
                if (
                    rank is not None
                    and agreement_type is not None
                    and count is not None
                    and count >= 0
                ):
                    rows.append(
                        {
                            "rank": int(rank),
                            "agreement_or_filing_type": agreement_type,
                            "document_count": int(count),
                        }
                    )
            projected = pd.DataFrame.from_records(
                rows,
                columns=[
                    "rank",
                    "agreement_or_filing_type",
                    "document_count",
                ],
            ).sort_values("rank", kind="stable", ignore_index=True)
            step.set_output(projected)

        tracker.record(
            "PROJECT(rows)",
            len(projected),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
