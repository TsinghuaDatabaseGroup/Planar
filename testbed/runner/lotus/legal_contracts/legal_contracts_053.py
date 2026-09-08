#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-053."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_iso_date,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-053"


def main():
    setup(max_tokens=512, task_prefix="CONTRACTEXHIBIT")
    tracker = StepTracker()

    with Timer() as timer:
        documents = load_document_corpus("contract-exhibit")
        tracker.record(
            "SCAN_DOCS(CONTRACTEXHIBIT, all)",
            None,
            len(documents),
            output=documents,
        )

        with tracker.step(
            "SEM_FILTER(insider-trading policy)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is an insider-trading policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(approval or adoption date)",
            input_rows=len(policies),
        ) as step:
            extracted = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "approval_date": (
                        "the policy approval or adoption date explicitly stated in "
                        "the document, formatted as YYYY-MM-DD; return null when none "
                        "is stated, and do not substitute a general effective date"
                    )
                },
            )
            extracted["approval_date"] = extracted["approval_date"].map(
                normalize_iso_date
            )
            extracted = extracted[
                ["document_id", "approval_date"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        dated = extracted.loc[extracted["approval_date"].notna()].reset_index(drop=True)
        tracker.record(
            "FILTER(approval_date IS NOT NULL)",
            len(extracted),
            len(dated),
            output=dated,
        )

        ordered = dated.sort_values(
            ["approval_date", "document_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(approval_date DESC, document_id ASC)",
            len(dated),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(1).reset_index(drop=True)
        tracker.record(
            "LIMIT(1)",
            len(ordered),
            len(limited),
            output=limited,
        )
        rows = df_records(limited)
        answer = rows[0] if rows else {}

    print(f"Result: {answer}")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
