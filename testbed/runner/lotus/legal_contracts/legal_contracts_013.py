#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-013."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    load_document_corpus,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-013"


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
            "SEM_FILTER(contractual acquisition-or-control standstill)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The agreement {text} contains a contractual acquisition-or-control "
                "standstill. Exclude insider-trading rules, descriptions of statutory "
                "anti-takeover law, securities-transfer lock-ups, instrument-exercise "
                "blockers, and no-shop duties."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(type, standstill duration, and governing law)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "agreement_type": "a concise stated agreement type",
                    "standstill_duration_years": (
                        "the contractual acquisition-or-control standstill duration "
                        "normalized to years, or null when unstated or unquantifiable"
                    ),
                    "governing_law": "the expressly stated governing-law jurisdiction",
                },
            )
            extracted["agreement_type"] = extracted["agreement_type"].map(clean_text)
            extracted["standstill_duration_years"] = extracted[
                "standstill_duration_years"
            ].map(parse_number)
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            step.set_output(extracted)

        qualifying = extracted.loc[
            extracted["standstill_duration_years"].notna()
            & (extracted["standstill_duration_years"] >= 2),
            [
                "document_id",
                "agreement_type",
                "standstill_duration_years",
                "governing_law",
            ],
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(standstill_duration_years >= 2)",
            len(extracted),
            len(qualifying),
            output=qualifying,
        )

        answer = df_records(qualifying)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
