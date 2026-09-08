#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-009."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-009"
COVERAGE_GROUPS = ("beyond_sec_minimum", "sec_minimum_only")


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
            "SEM_FILTER(EX-97 clawback or compensation recoupment policy)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is an EX-97 clawback or compensation recoupment "
                "policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(coverage group and separate misconduct trigger)",
            input_rows=len(policies),
        ) as step:
            extracted = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "coverage_group": (
                        "exactly beyond_sec_minimum if recovery extends beyond SEC "
                        "minimum requirements, otherwise sec_minimum_only"
                    ),
                    "misconduct_trigger": (
                        "true if employee misconduct is a recovery trigger separate "
                        "from an accounting restatement; false otherwise"
                    ),
                },
            )
            extracted["coverage_group"] = extracted["coverage_group"].map(
                lambda value: normalize_enum(value, COVERAGE_GROUPS)
            )
            extracted["misconduct_trigger"] = extracted["misconduct_trigger"].map(
                parse_bool
            )
            step.set_output(extracted)

        grouped = (
            extracted.groupby("coverage_group", sort=False, dropna=False)
            .agg(
                policy_count=("document_id", "size"),
                misconduct_fraction=("misconduct_trigger", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["misconduct_fraction"] = grouped[
                "misconduct_fraction"
            ].round(4)
        group_order = {
            group: index for index, group in enumerate(COVERAGE_GROUPS)
        }
        grouped = grouped.sort_values(
            "coverage_group",
            key=lambda values: values.map(group_order),
        ).reset_index(drop=True)
        tracker.record(
            "GROUP_BY([coverage_group], COUNT(*), ROUND(AVG(misconduct_trigger), 4))",
            len(extracted),
            len(grouped),
            output=grouped,
        )

        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
