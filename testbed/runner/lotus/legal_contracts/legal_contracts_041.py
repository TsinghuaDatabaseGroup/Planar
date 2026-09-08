#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-041."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    parse_optional_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-041"


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
            "SEM_FILTER(substantive insider-trading policy)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is a substantive insider-trading policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(covered persons and family-extension indicator)",
            input_rows=len(policies),
        ) as step:
            extracted = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "covered_persons": (
                        "a concise description of the categories of persons covered "
                        "by the trading restrictions"
                    ),
                    "extends_to_family": (
                        "true only if the trading restrictions explicitly extend to "
                        "covered persons' family or household members; false otherwise"
                    ),
                },
            )
            extracted["covered_persons"] = extracted["covered_persons"].map(clean_text)
            extracted["extends_to_family"] = extracted["extends_to_family"].map(
                parse_optional_bool
            )
            step.set_output(extracted)

        without_family = extracted.loc[
            extracted["extends_to_family"].eq(False)
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(extends_to_family = false)",
            len(extracted),
            len(without_family),
            output=without_family,
        )

        projected = without_family[
            ["document_id", "covered_persons"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(document_id, covered_persons)",
            len(without_family),
            len(projected),
            output=projected,
        )

        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
