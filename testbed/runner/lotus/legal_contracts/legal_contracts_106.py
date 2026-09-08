#!/usr/bin/env python3
"""LOTUS pipeline for legal_contracts-106."""

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
    parse_string_list,
    save_output,
    setup,
)

TASK_ID = "legal_contracts-106"
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
            "SCAN_DOCS(CONTRACTEXHIBIT, all) AS BASE",
            None,
            len(documents),
            output=documents,
        )
        with tracker.step(
            "SEM_FILTER(document has NDA status, including dual-status exhibits)",
            input_rows=len(documents),
        ) as step:
            ndas = documents.sem_filter(
                "The document {text} has non-disclosure-agreement status, including a "
                "dual-status SEC exhibit."
            ).reset_index(drop=True)
            step.set_output(ndas)

        with tracker.step(
            "SEM_EXTRACT(finite duration and governing-law jurisdictions)",
            input_rows=len(ndas),
        ) as step:
            base_records = ndas.sem_extract(
                input_cols=["text"],
                output_cols={
                    "duration_years": (
                        "the expressly stated finite confidentiality duration converted "
                        "to years; return null when missing or non-finite"
                    ),
                    "governing_law_jurisdictions": (
                        "a JSON list containing every expressly stated governing-law "
                        "jurisdiction"
                    ),
                },
            )
            base_records["duration_years"] = base_records["duration_years"].map(
                parse_number
            )
            base_records["governing_law_jurisdictions"] = base_records[
                "governing_law_jurisdictions"
            ].map(parse_string_list)
            step.set_output(base_records)

        finite_single_law = base_records[
            base_records["duration_years"].notna()
            & (base_records["governing_law_jurisdictions"].map(len) == 1)
        ].reset_index(drop=True)
        tracker.record(
            "FILTER(finite duration and exactly one jurisdiction)",
            len(base_records),
            len(finite_single_law),
            output=finite_single_law,
        )

        with tracker.step(
            "SEM_EXTRACT(exact five-position restriction profile)",
            input_rows=len(finite_single_law),
        ) as step:
            profiled = finite_single_law.sem_extract(
                input_cols=["text"],
                output_cols={
                    "has_non_compete": (
                        "true only if an operative non-compete clause is present"
                    ),
                    "has_customer_non_solicitation": (
                        "true only if an operative customer non-solicitation clause is "
                        "present"
                    ),
                    "has_employee_non_solicitation": (
                        "true only if an operative employee non-solicitation clause is "
                        "present"
                    ),
                    "has_standstill": (
                        "true only if an operative standstill clause is present"
                    ),
                    "has_vendor_non_solicitation": (
                        "true only if an operative vendor non-solicitation clause is "
                        "present"
                    ),
                },
            )
            for column in PROFILE_COLUMNS:
                profiled[column] = profiled[column].map(parse_bool)
            profiled["restriction_profile"] = [
                tuple(bool(row[column]) for column in PROFILE_COLUMNS)
                for _, row in profiled.iterrows()
            ]
            step.set_output(profiled)

        restrictive_mask = profiled["restriction_profile"].map(any).astype(bool)
        restrictive = profiled.loc[restrictive_mask].reset_index(drop=True)
        tracker.record(
            "FILTER(ANY_TRUE(restriction profile))",
            len(profiled),
            len(restrictive),
            output=restrictive,
        )

        eligible = restrictive[
            ["governing_law_jurisdictions", "restriction_profile", "duration_years"]
        ].copy()
        eligible["jurisdiction"] = eligible["governing_law_jurisdictions"].map(
            lambda values: values[0]
        )
        eligible = eligible[
            ["jurisdiction", "restriction_profile", "duration_years"]
        ].reset_index(drop=True)
        tracker.record(
            "PROJECT(jurisdiction, restriction profile, duration)",
            len(restrictive),
            len(eligible),
            output=eligible,
        )

        jurisdiction_counts = (
            eligible.groupby("jurisdiction", sort=False)
            .size()
            .rename("qualifying_nda_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(jurisdiction, qualifying NDA count)",
            len(eligible),
            len(jurisdiction_counts),
            output=jurisdiction_counts,
        )

        ordered_jurisdictions = jurisdiction_counts.sort_values(
            ["qualifying_nda_count", "jurisdiction"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(qualifying count DESC, jurisdiction ASC)",
            len(jurisdiction_counts),
            len(ordered_jurisdictions),
            output=ordered_jurisdictions,
        )

        top_jurisdictions = ordered_jurisdictions.head(3).reset_index(drop=True)
        tracker.record(
            "LIMIT(3 jurisdictions)",
            len(ordered_jurisdictions),
            len(top_jurisdictions),
            output=top_jurisdictions,
        )

        profiles = (
            eligible.groupby(
                ["jurisdiction", "restriction_profile"],
                sort=False,
            )
            .agg(
                profile_nda_count=("duration_years", "size"),
                profile_avg_duration_years=("duration_years", "mean"),
            )
            .reset_index()
        )
        if not profiles.empty:
            profiles["profile_avg_duration_years"] = profiles[
                "profile_avg_duration_years"
            ].round(2)
        tracker.record(
            "GROUP_BY(jurisdiction and profile, count and average duration)",
            len(eligible),
            len(profiles),
            output=profiles,
        )

        joined = top_jurisdictions.merge(
            profiles,
            on="jurisdiction",
            how="inner",
        )
        tracker.record(
            "JOIN(JURISDICTIONS.jurisdiction = PROFILES.jurisdiction)",
            {
                "JURISDICTIONS": len(top_jurisdictions),
                "PROFILES": len(profiles),
            },
            len(joined),
            output=joined,
        )

        ordered_profiles = joined.sort_values(
            [
                "qualifying_nda_count",
                "jurisdiction",
                "profile_nda_count",
                "restriction_profile",
            ],
            ascending=[False, True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(jurisdiction totals and profile rank)",
            len(joined),
            len(ordered_profiles),
            output=ordered_profiles,
        )

        most_common = ordered_profiles.drop_duplicates(
            subset=["jurisdiction"],
            keep="first",
        ).reset_index(drop=True)
        tracker.record(
            "DEDUP(jurisdiction, keep first profile)",
            len(ordered_profiles),
            len(most_common),
            output=most_common,
        )

        projected = most_common[
            [
                "jurisdiction",
                "qualifying_nda_count",
                "restriction_profile",
                "profile_nda_count",
                "profile_avg_duration_years",
            ]
        ].rename(
            columns={"restriction_profile": "most_common_restriction_profile"}
        )
        projected["most_common_restriction_profile"] = projected[
            "most_common_restriction_profile"
        ].map(list)
        tracker.record(
            "PROJECT(jurisdiction, total, most common profile and metrics)",
            len(most_common),
            len(projected),
            output=projected,
        )
        answer = df_records(projected)

    print(f"Result: {len(answer)} rows")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
