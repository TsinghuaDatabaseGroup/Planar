#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-070."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_optional_text,
    df_records,
    load_document_corpus,
    parse_bool,
    save_output,
    setup,
    stable_mode_optional,
)

TASK_ID = "legal_contracts-070"


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
            "SEM_FILTER(in-scope agreement with IP-ownership clause)",
            input_rows=len(documents),
        ) as step:
            agreements = documents.sem_filter(
                "The document {text} is a mutual non-disclosure agreement, a "
                "unilateral non-disclosure agreement, or a confidentiality-and-"
                "standstill agreement containing an intellectual-property ownership "
                "clause."
            ).reset_index(drop=True)
            step.set_output(agreements)

        with tracker.step(
            "SEM_EXTRACT(residual right, law, return requirement, and relief)",
            input_rows=len(agreements),
        ) as step:
            extracted = agreements.sem_extract(
                input_cols=["text"],
                output_cols={
                    "residual_information_present": (
                        "true if the agreement grants a residual-information right; "
                        "false otherwise"
                    ),
                    "governing_law": (
                        "the expressly stated governing-law jurisdiction, or null "
                        "when unstated"
                    ),
                    "return_of_materials": (
                        "true if return or destruction of confidential materials is "
                        "required; false otherwise"
                    ),
                    "injunctive_relief": (
                        "true if injunctive or equitable relief is provided; false "
                        "otherwise"
                    ),
                },
            )
            for column in (
                "residual_information_present",
                "return_of_materials",
                "injunctive_relief",
            ):
                extracted[column] = extracted[column].map(parse_bool)
            extracted["governing_law"] = extracted["governing_law"].map(
                clean_optional_text
            )
            step.set_output(extracted)

        projected = extracted[
            [
                "residual_information_present",
                "governing_law",
                "return_of_materials",
                "injunctive_relief",
            ]
        ].rename(
            columns={
                "return_of_materials": "return_flag",
                "injunctive_relief": "injunctive_flag",
            }
        )
        projected["return_flag"] = projected["return_flag"].astype(int)
        projected["injunctive_flag"] = projected["injunctive_flag"].astype(int)
        projected = projected.reset_index(drop=True)
        tracker.record(
            "PROJECT(residual flag, law, return flag, injunctive flag)",
            len(extracted),
            len(projected),
            output=projected,
        )

        grouped = (
            projected.groupby("residual_information_present", sort=False)
            .agg(
                agreement_count=("residual_information_present", "size"),
                most_common_governing_law=(
                    "governing_law",
                    stable_mode_optional,
                ),
                return_of_materials_rate=("return_flag", "mean"),
                injunctive_relief_rate=("injunctive_flag", "mean"),
            )
            .reset_index()
        )
        grouped["return_of_materials_rate"] = grouped[
            "return_of_materials_rate"
        ].round(4)
        grouped["injunctive_relief_rate"] = grouped[
            "injunctive_relief_rate"
        ].round(4)
        tracker.record(
            "GROUP_BY([residual flag], count, law mode, return rate, relief rate)",
            len(projected),
            len(grouped),
            output=grouped,
        )
        answer = df_records(grouped)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
