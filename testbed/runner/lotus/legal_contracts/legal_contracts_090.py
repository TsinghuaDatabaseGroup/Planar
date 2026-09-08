#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-090."""

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

TASK_ID = "legal_contracts-090"
EXCEPTION_COLUMNS = (
    "public_information",
    "prior_knowledge",
    "independent_development",
    "third_party_receipt",
    "legal_compulsion",
    "written_consent",
)


def _comparison(row):
    employee = row["employee_non_solicitation_duration"]
    standstill = row["standstill_duration"]
    if employee > standstill:
        return "employee_restriction_longer"
    if employee == standstill:
        return "equal"
    return "standstill_longer"


def main():
    setup(max_tokens=1024, task_prefix="CONTRACTEXHIBIT")
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
            "SEM_FILTER(NDA including dual-labelled NDA exhibit)",
            input_rows=len(documents),
        ) as step:
            ndas = documents.sem_filter(
                "The document {text} is a non-disclosure agreement, including a "
                "dual-labelled NDA exhibit."
            ).reset_index(drop=True)
            step.set_output(ndas)

        with tracker.step(
            "SEM_FILTER(standstill and employee non-solicitation clauses)",
            input_rows=len(ndas),
        ) as step:
            dual_restrictions = ndas.sem_filter(
                "The agreement {text} expressly contains both an operative standstill "
                "provision and an operative employee non-solicitation clause."
            ).reset_index(drop=True)
            step.set_output(dual_restrictions)

        with tracker.step(
            "SEM_EXTRACT(restriction durations and six exception flags)",
            input_rows=len(dual_restrictions),
        ) as step:
            extracted = dual_restrictions.sem_extract(
                input_cols=["text"],
                output_cols={
                    "standstill_duration": (
                        "the standstill duration normalized to months, or null when "
                        "unstated or unquantifiable"
                    ),
                    "employee_non_solicitation_duration": (
                        "the employee-non-solicitation duration normalized to months, "
                        "or null when unstated or unquantifiable"
                    ),
                    "public_information": (
                        "true if an operative public-information exception is present"
                    ),
                    "prior_knowledge": (
                        "true if an operative prior-knowledge exception is present"
                    ),
                    "independent_development": (
                        "true if an operative independent-development exception is present"
                    ),
                    "third_party_receipt": (
                        "true if an operative lawful third-party-receipt exception is present"
                    ),
                    "legal_compulsion": (
                        "true if an operative legal-compulsion exception is present"
                    ),
                    "written_consent": (
                        "true if an operative written-consent exception is present"
                    ),
                },
            )
            extracted["standstill_duration"] = extracted[
                "standstill_duration"
            ].map(parse_number)
            extracted["employee_non_solicitation_duration"] = extracted[
                "employee_non_solicitation_duration"
            ].map(parse_number)
            for column in EXCEPTION_COLUMNS:
                extracted[column] = extracted[column].map(parse_bool)
            step.set_output(extracted)

        complete = extracted.loc[
            extracted["standstill_duration"].notna()
            & extracted["employee_non_solicitation_duration"].notna()
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(both restriction durations IS NOT NULL)",
            len(extracted),
            len(complete),
            output=complete,
        )

        projected = complete[list(EXCEPTION_COLUMNS)].sum(axis=1).to_frame(
            "exception_count"
        )
        projected["class_label"] = complete.apply(_comparison, axis=1).values
        projected = projected[["class_label", "exception_count"]].reset_index(
            drop=True
        )
        tracker.record(
            "PROJECT(duration comparison class, exception count)",
            len(complete),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("class_label", sort=False)
            .agg(
                agreement_count=("class_label", "size"),
                average_exception_count=("exception_count", "mean"),
            )
            .reset_index()
        )
        if not grouped.empty:
            grouped["average_exception_count"] = grouped[
                "average_exception_count"
            ].round(2)
        tracker.record(
            "GROUP_BY([class_label], count, average exception count)",
            len(projected),
            len(grouped),
            output=grouped,
        )

        ordered = grouped.sort_values(
            ["agreement_count", "class_label"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(agreement_count DESC, class_label ASC)",
            len(grouped),
            len(ordered),
            output=ordered,
        )
        answer = df_records(ordered)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
