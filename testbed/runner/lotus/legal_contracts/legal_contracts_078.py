#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-078."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-078"


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

        long_documents = documents.loc[
            documents["word_count"] > 5000
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(word_count > 5000)",
            len(documents),
            len(long_documents),
            output=long_documents,
        )

        with tracker.step(
            "SEM_FILTER(substantive insider-trading policy)",
            input_rows=len(long_documents),
        ) as step:
            policies = long_documents.sem_filter(
                "The document {text} is a substantive insider-trading policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_FILTER(Rule 10b5-1 plans and family coverage)",
            input_rows=len(policies),
        ) as step:
            covered = policies.sem_filter(
                "The policy {text} discusses Rule 10b5-1 plans and expressly covers "
                "family or household members of covered persons. Both conditions "
                "must hold."
            ).reset_index(drop=True)
            step.set_output(covered)

        with tracker.step(
            "SEM_FILTER(no policy approval or adoption date)",
            input_rows=len(covered),
        ) as step:
            undated = covered.sem_filter(
                "The policy {text} states no policy approval or adoption date. A "
                "general effective date is not an approval or adoption date."
            ).reset_index(drop=True)
            step.set_output(undated)

        with tracker.step(
            "SEM_EXTRACT(company names)",
            input_rows=len(undated),
        ) as step:
            extracted = undated.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company_names": (
                        "a list of all company names stated in the policy; exclude "
                        "individual officer names"
                    )
                },
            )
            extracted["company_names"] = extracted["company_names"].map(
                parse_string_list
            )
            extracted = extracted[
                ["document_id", "company_names"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        answer = df_records(extracted)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
