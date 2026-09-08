#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-094."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    normalize_enum,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-094"


def main():
    setup(max_tokens=8192, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(non-disclosure agreement, including dual-labeled exhibits)",
            input_rows=len(documents),
        ) as step:
            nda_documents = documents.sem_filter(
                "The document {text} is a non-disclosure agreement, including an "
                "exhibit that is dual-labeled as an NDA and another agreement type."
            ).reset_index(drop=True)
            step.set_output(nda_documents)

        with tracker.step(
            "SEM_FILTER(definition, return-or-destroy, and injunctive relief)",
            input_rows=len(nda_documents),
        ) as step:
            qualifying = nda_documents.sem_filter(
                "The agreement {text} explicitly defines confidential information "
                "and contains both an operative return-or-destroy obligation and "
                "injunctive-relief language."
            ).reset_index(drop=True)
            step.set_output(qualifying)

        with tracker.step(
            "SEM_EXTRACT(governing law, NDA direction, and finite term)",
            input_rows=len(qualifying),
        ) as step:
            extracted = qualifying.sem_extract(
                input_cols=["text"],
                output_cols={
                    "governing_law": (
                        "the normalized governing-law jurisdiction; use a consistent "
                        "full jurisdiction name, or null if none is stated"
                    ),
                    "nda_direction": (
                        "mutual if both parties owe confidentiality duties, unilateral "
                        "if only one party does, otherwise null"
                    ),
                    "finite_term_years": (
                        "the explicitly stated finite agreement term converted to "
                        "years; return null for perpetual or unspecified terms"
                    ),
                },
            )
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            extracted["nda_direction"] = extracted["nda_direction"].map(
                lambda value: normalize_enum(value, ["mutual", "unilateral"])
            )
            extracted["finite_term_years"] = extracted["finite_term_years"].map(
                parse_number
            )
            extracted = extracted[
                ["governing_law", "nda_direction", "finite_term_years"]
            ].reset_index(drop=True)
            step.set_output(extracted)

        filtered = extracted[
            extracted["governing_law"].notna()
            & extracted["nda_direction"].isin(["mutual", "unilateral"])
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(governing law present and direction valid)",
            len(extracted),
            len(filtered),
            output=filtered,
        )

        grouped = (
            filtered.groupby("governing_law", sort=False)
            .agg(
                mutual_count=(
                    "nda_direction",
                    lambda values: int((values == "mutual").sum()),
                ),
                unilateral_count=(
                    "nda_direction",
                    lambda values: int((values == "unilateral").sum()),
                ),
                average_finite_term_years=("finite_term_years", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["average_finite_term_years"] = grouped[
                "average_finite_term_years"
            ].round(2)
        tracker.record(
            "GROUP_BY(governing law, direction counts, average finite term)",
            len(filtered),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.assign(
            combined_count=grouped["mutual_count"] + grouped["unilateral_count"]
        ).sort_values(
            ["combined_count", "governing_law"],
            ascending=[False, True],
            kind="stable",
        )
        ordered = ordered.drop(columns=["combined_count"]).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(combined count DESC, governing law ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(3).reset_index(drop=True)
        tracker.record(
            "LIMIT(3)",
            len(ordered),
            len(limited),
            output=limited,
        )
        answer = df_records(limited)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
