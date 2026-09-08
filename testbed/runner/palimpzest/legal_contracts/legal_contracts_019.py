#!/usr/bin/env python3
"""Palimpzest operator-isolation pipeline for legal_contracts-019."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    DATA_ROOT,
    StepTracker,
    Timer,
    df_records,
    get_config,
    load_html_documents,
    normalize_enum,
    parse_record_list,
    run_bounded_semantic_aggregate,
    save_output,
)

TASK_ID = "legal_contracts-019"
OPTIMIZER = "pareto (multi-model static estimates; no native aggregate validator)"
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

        aggregate = run_bounded_semantic_aggregate(
            TASK_ID,
            reports,
            config,
            tracker=tracker,
            col={
                "name": "rows",
                "type": list[dict],
                "desc": (
                    "Five ranked objects, each with exactly the keys \"rank\" "
                    "(integer), \"agreement_or_filing_type\", and "
                    "\"document_count\" (integer)."
                ),
            },
            agg=(
                "Across the complete input relation, assign each document one "
                "primary agreement or filing type and return the five most "
                "common types with their distinct document counts. Allowed "
                "types are sox_certification, employment_agreement, "
                "insider_trading_policy, auditor_consent, subsidiary_list, "
                "clawback_policy, mutual_nda, unilateral_nda, "
                "confidentiality_standstill, non_compete_agreement, "
                "supply_agreement, non_solicitation_agreement, and other. Treat "
                "genuine Section 302 or 906 officer certifications as "
                "sox_certification, but Regulation AB servicing certifications "
                "as other. Rank by document count descending and then category "
                "alphabetically."
            ),
            depends_on=["filename", "text"],
            partial_col={
                "name": "partial_rows",
                "type": list[dict],
                "desc": (
                    "Thirteen additive sufficient-statistic objects, each "
                    "with exactly the keys \"agreement_or_filing_type\" and "
                    "\"document_count\" (integer)."
                ),
            },
            partial_agg=(
                "Within only this input chunk, assign every distinct document "
                "exactly one primary agreement or filing type and return "
                "additive document counts for all allowed types, including zero "
                "counts. Allowed types are sox_certification, "
                "employment_agreement, insider_trading_policy, auditor_consent, "
                "subsidiary_list, clawback_policy, mutual_nda, unilateral_nda, "
                "confidentiality_standstill, non_compete_agreement, "
                "supply_agreement, non_solicitation_agreement, and other. Treat "
                "genuine Section 302 or 906 officer certifications as "
                "sox_certification, but Regulation AB servicing certifications "
                "as other."
            ),
            partial_depends_on=[
                "agreement_or_filing_type",
                "document_count",
            ],
            partial_merge_agg=(
                "These inputs are additive sufficient statistics from disjoint "
                "document chunks. Sum document_count for each of the thirteen "
                "agreement_or_filing_type labels and return all labels, "
                "including zero counts, in the same sufficient-statistic format."
            ),
            merge_agg=(
                "These inputs are additive sufficient statistics from disjoint "
                "chunks of the complete relation. Sum document_count for each "
                "agreement_or_filing_type and return the five types with the "
                "largest positive document counts. Rank by document_count "
                "descending and then category alphabetically."
            ),
            merge_max_rows=50,
            merge_sort_by=["agreement_or_filing_type"],
            final_max_rows=100,
        )
        aggregate_frame = aggregate.output
        if aggregate_frame.empty:
            raise ValueError(f"{TASK_ID}: semantic aggregate returned no answer")

        projected_rows = []
        seen_types = set()
        for item in parse_record_list(aggregate_frame.iloc[0]["rows"]):
            rank = int(item.get("rank"))
            agreement_type = normalize_enum(
                item.get("agreement_or_filing_type"),
                AGREEMENT_TYPES,
            )
            document_count = int(item.get("document_count"))
            if rank < 1 or agreement_type is None or agreement_type in seen_types or document_count < 1:
                raise ValueError(f"{TASK_ID}: invalid aggregate row")
            seen_types.add(agreement_type)
            projected_rows.append(
                {
                    "rank": rank,
                    "agreement_or_filing_type": agreement_type,
                    "document_count": document_count,
                }
            )
        projected = (
            pd.DataFrame.from_records(
                projected_rows,
                columns=[
                    "rank",
                    "agreement_or_filing_type",
                    "document_count",
                ],
            )
            .sort_values("rank")
            .reset_index(drop=True)
        )
        if len(projected) > 5 or projected["rank"].duplicated().any():
            raise ValueError(f"{TASK_ID}: invalid aggregate ranking")
        tracker.record("project", len(aggregate_frame), projected)
        answer = df_records(projected)

    save_output(
        TASK_ID,
        answer,
        timer.elapsed + aggregate.resumed_elapsed_seconds,
        tracker,
    )


if __name__ == "__main__":
    main()
