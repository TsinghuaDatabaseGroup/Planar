#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-079."""

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

TASK_ID = "legal_contracts-079"


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
            "SEM_FILTER(NDA or confidentiality-and-standstill agreement)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(explicit effective date)",
            input_rows=len(agreements),
        ) as step:
            dated = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "effective_date": (
                        "the explicitly stated effective date formatted as YYYY-MM-DD, "
                        "or null when no effective date is stated"
                    )
                },
            )
            dated["effective_date"] = dated["effective_date"].map(normalize_iso_date)
            step.set_output(dated)

        after_2010 = dated.loc[
            dated["effective_date"].notna()
            & (dated["effective_date"] > "2010-12-31")
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(effective_date > DATE('2010-12-31'))",
            len(dated),
            len(after_2010),
            output=after_2010,
        )

        with tracker.step(
            "SEM_FILTER(protected information includes computer-related material)",
            input_rows=len(after_2010),
        ) as step:
            electronic = after_2010.sem_filter(
                "The protected-information definition in {text} explicitly includes "
                "electronic, digital, computer, software, or comparable computer-"
                "related material."
            ).reset_index(drop=True)
            step.set_output(electronic)

        with tracker.step(
            "SEM_FILTER(confidentiality survives termination or disposition)",
            input_rows=len(electronic),
        ) as step:
            surviving = electronic.sem_filter(
                "The confidentiality obligations in {text} expressly continue after "
                "agreement termination or after protected material is returned or "
                "destroyed."
            ).reset_index(drop=True)
            step.set_output(surviving)

        with tracker.step(
            "SEM_FILTER(destruction available instead of return)",
            input_rows=len(surviving),
        ) as step:
            destroyable = surviving.sem_filter(
                "Under {text}, the recipient may choose, or may be directed, to destroy "
                "protected material instead of returning that same material."
            ).reset_index(drop=True)
            step.set_output(destroyable)

        with tracker.step(
            "SEM_EXTRACT(governing law and single archival-copy permission)",
            input_rows=len(destroyable),
        ) as step:
            extracted = destroyable.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": "the expressly stated governing-law jurisdiction",
                    "allows_single_archival_copy": (
                        "true only if the agreement expressly permits retention of "
                        "one single archival or record copy; exclude automatic-backup "
                        "exceptions and permissions to retain an unspecified number "
                        "of copies"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(clean_text)
            extracted["allows_single_archival_copy"] = extracted[
                "allows_single_archival_copy"
            ].map(parse_optional_bool)
            step.set_output(extracted)

        projected = extracted[
            ["document_id", "governing_law", "allows_single_archival_copy"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(document_id, governing_law, allows_single_archival_copy)",
            len(extracted),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
