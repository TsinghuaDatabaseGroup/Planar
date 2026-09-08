#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-074."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_optional_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-074"
FEATURE_COLUMNS = (
    "blackout_period",
    "pre_clearance",
    "rule_10b5_1",
    "family_household_coverage",
    "pledging_prohibited",
)


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
            "SEM_FILTER(insider-trading policy)",
            input_rows=len(documents),
        ) as step:
            policies = documents.sem_filter(
                "The document {text} is an insider-trading policy."
            ).reset_index(drop=True)
            step.set_output(policies)

        with tracker.step(
            "SEM_EXTRACT(five insider-trading policy flags)",
            input_rows=len(policies),
        ) as step:
            extracted = policies.sem_extract(
                input_cols=["text"],
                output_cols={
                    "blackout_period": (
                        "true if the policy has a blackout period, false if it "
                        "definitely does not, or null when undetermined"
                    ),
                    "pre_clearance": (
                        "true if the policy requires pre-clearance, false if it "
                        "definitely does not, or null when undetermined"
                    ),
                    "rule_10b5_1": (
                        "true if the policy discusses Rule 10b5-1 plans, false if it "
                        "definitely does not, or null when undetermined"
                    ),
                    "family_household_coverage": (
                        "true if the policy covers family or household members, false "
                        "if it definitely does not, or null when undetermined"
                    ),
                    "pledging_prohibited": (
                        "true if the policy prohibits pledging, false if it definitely "
                        "does not, or null when undetermined"
                    ),
                },
            )
            for column in FEATURE_COLUMNS:
                extracted[column] = extracted[column].map(parse_optional_bool)
            step.set_output(extracted)

        complete = extracted.dropna(subset=list(FEATURE_COLUMNS)).copy()
        for column in FEATURE_COLUMNS:
            complete[column] = complete[column].astype(bool)
        complete = complete.reset_index(drop=True)
        tracker.record(
            "FILTER(all five policy flags IS NOT NULL)",
            len(extracted),
            len(complete),
            output=complete,
        )

        grouped = (
            complete.groupby(list(FEATURE_COLUMNS), sort=False)
            .size()
            .rename("policy_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(five policy flags, COUNT(*))",
            len(complete),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["policy_count", *FEATURE_COLUMNS],
            ascending=[False, True, True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(policy_count DESC, flags ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
