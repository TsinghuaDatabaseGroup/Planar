#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-073."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    clean_text,
    df_records,
    load_document_corpus,
    parse_optional_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-073"


def main():
    setup(max_tokens=768, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(clawback or compensation-recovery policy)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is a clawback or compensation-recovery policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(policy triggers, lookback, and expanded coverage)",
            input_rows=len(policies),
        ) as step:
            extracted = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "document_name": "the stated document name",
                    "policy_title": "the stated policy title",
                    "lookback_period": "the lookback period stated by the policy",
                    "restatement_trigger": (
                        "true if a financial restatement triggers recovery; false otherwise"
                    ),
                    "misconduct_trigger": (
                        "true if misconduct triggers recovery; false otherwise"
                    ),
                    "expanded_covered_group": (
                        "a concise description of the covered group beyond current or "
                        "former Rule 10D-1 or Section 16 executive officers; return "
                        "null if coverage does not extend beyond those officers"
                    ),
                },
            )
            for column in ("document_name", "policy_title", "lookback_period"):
                extracted[column] = extracted[column].map(clean_text)
            extracted["restatement_trigger"] = extracted[
                "restatement_trigger"
            ].map(parse_optional_bool)
            extracted["misconduct_trigger"] = extracted["misconduct_trigger"].map(
                parse_optional_bool
            )
            extracted["expanded_covered_group"] = extracted[
                "expanded_covered_group"
            ].map(clean_optional_text)
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["restatement_trigger"].eq(True)
            & extracted["misconduct_trigger"].eq(True)
            & extracted["expanded_covered_group"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(restatement_trigger AND misconduct_trigger AND expanded group IS NOT NULL)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        projected = qualifying[
            [
                "document_name",
                "policy_title",
                "lookback_period",
                "expanded_covered_group",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(document name, title, lookback, expanded group)",
            len(qualifying),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
