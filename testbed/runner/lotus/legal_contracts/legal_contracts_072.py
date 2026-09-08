#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-072."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    normalize_iso_date,
    parse_optional_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-072"


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
            "SEM_FILTER(insider-trading policy covering family or household)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is an insider-trading policy that explicitly "
                "covers family or household members of covered persons."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(policy features and date)",
            input_rows=len(policies),
        ) as step:
            extracted = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "document_name": "the stated document name",
                    "policy_title": "the stated policy title",
                    "discusses_10b5_1": (
                        "true if the policy discusses Rule 10b5-1 plans; false otherwise"
                    ),
                    "prohibits_short_sales": (
                        "true if the policy prohibits short sales; false otherwise"
                    ),
                    "prohibits_pledging": (
                        "true if the policy prohibits pledging; false otherwise"
                    ),
                    "policy_date": (
                        "the stated policy approval or effective date formatted as "
                        "YYYY-MM-DD, or null when neither is stated"
                    ),
                },
            )
            extracted["document_name"] = extracted["document_name"].map(clean_text)
            extracted["policy_title"] = extracted["policy_title"].map(clean_text)
            for column in (
                "discusses_10b5_1",
                "prohibits_short_sales",
                "prohibits_pledging",
            ):
                extracted[column] = extracted[column].map(parse_optional_bool)
            extracted["policy_date"] = extracted["policy_date"].map(
                normalize_iso_date
            )
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["discusses_10b5_1"].eq(True)
            & extracted["prohibits_short_sales"].eq(True)
            & extracted["prohibits_pledging"].eq(True)
            & extracted["policy_date"].notna()
            & (extracted["policy_date"] < "2024-01-01"),
            ["document_name", "policy_title", "policy_date"],
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(10b5-1, short-sale, pledging flags true and policy date before 2024)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        ordered = qualifying.sort_values(
            ["policy_date", "document_name"],
            ascending=[True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(policy_date ASC, document_name ASC)",
            len(qualifying),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
