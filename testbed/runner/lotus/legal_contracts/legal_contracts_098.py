#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-098."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    df_records,
    load_document_corpus,
    parse_bool,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-098"
PROFILE_COLUMNS = [
    "has_non_compete",
    "has_customer_non_solicitation",
    "has_employee_non_solicitation",
    "has_standstill",
    "has_vendor_non_solicitation",
]


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
            "SEM_FILTER(document has NDA status, including dual-labeled exhibits)",
            input_rows=len(documents),
        ) as step:
            nda_documents = documents.sem_filter(
                "The document {text} has non-disclosure-agreement status, including "
                "documents dual-labeled as an NDA and another SEC exhibit type."
            ).reset_index(drop=True)
            step.set_output(nda_documents)

        with tracker.step(
            "SEM_EXTRACT(finite confidentiality duration in years)",
            input_rows=len(nda_documents),
        ) as step:
            durations = nda_documents.sem_extract(
                input_cols=["text"],
                output_cols={
                    "duration_years": (
                        "the expressly stated finite confidentiality duration converted "
                        "to years; return null when the term is missing or non-finite"
                    )
                },
            )
            durations["duration_years"] = durations["duration_years"].map(
                parse_number
            )
            step.set_output(durations)

        finite_durations = durations[
            durations["duration_years"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(duration_years IS NOT NULL)",
            len(durations),
            len(finite_durations),
            output=finite_durations,
        )

        with tracker.step(
            "SEM_EXTRACT(exact five-position restriction profile)",
            input_rows=len(finite_durations),
        ) as step:
            profiled = finite_durations.sem_extract(
                input_cols=["text"],
                output_cols={
                    "has_non_compete": (
                        "true only if the NDA contains an operative non-compete clause"
                    ),
                    "has_customer_non_solicitation": (
                        "true only if the NDA contains an operative customer "
                        "non-solicitation clause"
                    ),
                    "has_employee_non_solicitation": (
                        "true only if the NDA contains an operative employee "
                        "non-solicitation clause"
                    ),
                    "has_standstill": (
                        "true only if the NDA contains an operative standstill clause"
                    ),
                    "has_vendor_non_solicitation": (
                        "true only if the NDA contains an operative vendor "
                        "non-solicitation clause"
                    ),
                },
            )
            for column in PROFILE_COLUMNS:
                profiled[column] = profiled[column].map(parse_bool)
            profiled["restriction_profile"] = [
                tuple(bool(row[column]) for column in PROFILE_COLUMNS)
                for _, row in profiled.iterrows()
            ]
            profiled = profiled[
                ["restriction_profile", "duration_years"]
            ].reset_index(drop=True)
            step.set_output(profiled)

        grouped = (
            profiled.groupby("restriction_profile", sort=False)
            .agg(
                member_count=("duration_years", "size"),
                avg_duration_years=("duration_years", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["avg_duration_years"] = grouped[
                "avg_duration_years"
            ].round(2)
        grouped["restriction_profile"] = grouped["restriction_profile"].map(list)
        tracker.record(
            "GROUP_BY(restriction profile, count, average duration)",
            len(profiled),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            "member_count",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(member_count DESC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )

        limited = ordered.head(5).reset_index(drop=True)
        tracker.record(
            "LIMIT(5)",
            len(ordered),
            len(limited),
            output=limited,
        )
        answer = df_records(limited)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
