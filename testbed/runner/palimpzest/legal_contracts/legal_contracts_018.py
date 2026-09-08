#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for legal_contracts-018."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    DATA_ROOT,
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_html_documents,
    memory_dataset,
    normalize_enum,
    result_frame,
    run_optimized_plan,
    save_output,
)

TASK_ID = "legal_contracts-018"
OPTIMIZER = "pareto + sentinel/mab (current-input sampling; 397B LLM validator)"
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


def main() -> None:
    config = get_config(
        max_tokens=4096,
        all_optimizations=True,
        include_small_model=True,
    )
    tracker = StepTracker(TASK_ID, optimizer_strategy=OPTIMIZER)

    with Timer() as timer:
        document_root = DATA_ROOT / "contract-exhibit"
        documents = pd.DataFrame.from_records(
            [
                {
                    "contract_id": path.name,
                    "doc_format": path.suffix.lower().lstrip("."),
                }
                for path in sorted(document_root.iterdir())
                if path.is_file()
            ]
        )
        tracker.record("scan", None, documents)
        html_documents = documents.loc[documents["doc_format"] == "htm"].reset_index(drop=True)
        tracker.record("filter", len(documents), html_documents)

        reports = load_html_documents(
            "contract-exhibit",
            filenames=html_documents["contract_id"],
        )
        tracker.record("scan", len(html_documents), reports)

        plan = memory_dataset(TASK_ID, reports).sem_map(
            cols=[
                {
                    "name": "agreement_or_filing_type",
                    "type": str,
                    "desc": (
                        "Exactly one of sox_certification, employment_agreement, "
                        "insider_trading_policy, auditor_consent, subsidiary_list, "
                        "clawback_policy, mutual_nda, unilateral_nda, "
                        "confidentiality_standstill, non_compete_agreement, "
                        "supply_agreement, non_solicitation_agreement, or other. "
                        "Genuine Section 302 or 906 officer certifications are "
                        "sox_certification; Regulation AB servicing "
                        "certifications are other."
                    ),
                }
            ],
            desc="Assign exactly one primary agreement or filing type.",
            depends_on=["filename", "text"],
        )
        started = time.time()
        semantic_result = run_optimized_plan(plan, config)
        classified = result_frame(
            semantic_result,
            generated_columns=["agreement_or_filing_type"],
        )
        classified["agreement_or_filing_type"] = classified["agreement_or_filing_type"].map(
            lambda value: normalize_enum(value, AGREEMENT_TYPES)
        )
        classified["agreement_or_filing_type"] = classified["agreement_or_filing_type"].fillna("other")
        classified = classified.rename(columns={"filename": "contract_id"})
        tracker.record_semantic(
            "sem_map",
            len(html_documents),
            classified[["contract_id", "agreement_or_filing_type"]],
            semantic_result,
            time.time() - started,
        )

        grouped = (
            classified.groupby(
                "agreement_or_filing_type",
                as_index=False,
            )["contract_id"]
            .nunique()
            .rename(columns={"contract_id": "document_count"})
            .sort_values(
                ["document_count", "agreement_or_filing_type"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )
        tracker.record("groupby", len(classified), grouped)
        tracker.record("order_by", len(grouped), grouped)
        answer = df_records(grouped)

    save_output(TASK_ID, answer, timer.elapsed, tracker)


if __name__ == "__main__":
    main()
