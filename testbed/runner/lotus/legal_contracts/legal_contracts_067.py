#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-067."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    normalize_company_name,
    normalize_enum,
    normalize_iso_date,
    parse_optional_bool,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-067"
TRIGGER_PATTERNS = ("restatement_only", "restatement_and_misconduct")


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
            "SEM_FILTER(beyond baseline SEC requirements with identified company)",
            input_rows=len(policies),
        ) as step:
            qualifying = policies.sem_filter(
                "The policy {text} goes beyond baseline SEC clawback requirements and "
                "identifies the company whose policy it is. Both conditions must hold."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(company, version, trigger pattern, and coverage breadth)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "company": "the primary company name stated in the policy",
                    "policy_version_date": (
                        "the latest identifiable policy version, approval, or adoption "
                        "date formatted as YYYY-MM-DD, or null when unidentifiable"
                    ),
                    "trigger_pattern": (
                        "exactly restatement_only or restatement_and_misconduct"
                    ),
                    "coverage_extends_beyond_executive_officers": (
                        "true if covered persons extend beyond executive officers; "
                        "false otherwise"
                    ),
                },
            ).rename(columns={"document_id": "policy_document_id"})
            extracted["company"] = extracted["company"].map(clean_text)
            extracted["company_key"] = extracted["company"].map(
                normalize_company_name
            )
            extracted["policy_version_date"] = extracted[
                "policy_version_date"
            ].map(normalize_iso_date)
            extracted["trigger_pattern"] = extracted["trigger_pattern"].map(
                lambda value: normalize_enum(value, TRIGGER_PATTERNS)
            )
            extracted["coverage_extends_beyond_executive_officers"] = extracted[
                "coverage_extends_beyond_executive_officers"
            ].map(parse_optional_bool)
            step.set_output(extracted)

        latest = (
            extracted.sort_values(
                ["company_key", "policy_version_date", "policy_document_id"],
                ascending=[True, False, True],
                na_position="last",
                kind="stable",
            )
            .drop_duplicates("company_key", keep="first")
            .reset_index(drop=True)
        )
        tracker.record(
            "DEDUP(normalized company, keep latest version then document ID ASC)",
            len(extracted),
            len(latest),
            output=latest,
        )

        projected = latest[
            [
                "company",
                "trigger_pattern",
                "coverage_extends_beyond_executive_officers",
            ]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(company, trigger_pattern, coverage breadth)",
            len(latest),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
