#!/usr/bin/env python3
"""LOTUS pipeline for finance-031."""

import os
import sys
from decimal import ROUND_HALF_UP, Decimal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from pipeline_helpers import (  # noqa: E402
    StepTracker,
    Timer,
    clean_text,
    df_records,
    iter_selected_texts,
    load_selected_texts,
    load_table,
    parse_number,
    save_output,
    setup,
)

TASK_ID = "finance-031"
BATCH_SIZE = 100


def _mean_change(adopters: pd.DataFrame) -> float | None:
    complete = adopters.dropna(
        subset=["risk_factor_count_2019", "risk_factor_count_2020"]
    )
    if complete.empty:
        return None
    total = sum(
        Decimal(str(current)) - Decimal(str(previous))
        for previous, current in zip(
            complete["risk_factor_count_2019"],
            complete["risk_factor_count_2020"],
        )
    )
    return float(
        (total / Decimal(len(complete))).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
    )


def main():
    setup(max_tokens=4096)
    tracker = StepTracker()

    with Timer() as timer:
        filings_2019 = load_table("SEC", "CSV/2019.csv")[["cik", "text"]].rename(
            columns={"text": "text_2019"}
        )
        filings_2019["cik"] = pd.to_numeric(
            filings_2019["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2019.csv AS filings_2019)",
            None,
            len(filings_2019),
            output=filings_2019,
        )
        tracker.record(
            "SCAN_DOCS(selector=filings_2019.text)",
            len(filings_2019),
            len(filings_2019),
            output=filings_2019,
        )

        filings_2020 = load_table("SEC", "CSV/2020.csv")[["cik", "text"]].rename(
            columns={"text": "text_2020"}
        )
        filings_2020["cik"] = pd.to_numeric(
            filings_2020["cik"], errors="coerce"
        ).astype("Int64")
        tracker.record(
            "SCAN_TABLE(CSV/2020.csv AS filings_2020)",
            None,
            len(filings_2020),
            output=filings_2020,
        )
        tracker.record(
            "SCAN_DOCS(selector=filings_2020.text)",
            len(filings_2020),
            len(filings_2020),
            output=filings_2020,
        )

        paired = filings_2019.merge(
            filings_2020,
            on="cik",
            how="inner",
            sort=False,
        )
        tracker.record(
            "JOIN(inner, filings_2019.cik = filings_2020.cik)",
            {"left": len(filings_2019), "right": len(filings_2020)},
            len(paired),
            output=paired,
        )

        with tracker.step(
            "SEM_FILTER(2020 pandemic or public-health risk factor)",
            input_rows=len(paired),
        ) as step:
            pandemic_parts = []
            for documents in iter_selected_texts(
                "SEC",
                paired,
                path_column="text_2020",
                output_column="document_2020",
                batch_size=BATCH_SIZE,
            ):
                selected = documents.sem_filter(
                    "The 2020 10-K filing {document_2020} includes a pandemic or "
                    "public-health risk factor for the reporting company; the "
                    "relevant disclosure is presented as a risk factor in the filing."
                )
                pandemic_parts.append(
                    selected.drop(columns=["document_2020"], errors="ignore")
                )
            pandemic_2020 = (
                pd.concat(pandemic_parts, ignore_index=True)
                if pandemic_parts
                else paired.iloc[0:0].copy()
            )
            step.set_output(pandemic_2020)

        with tracker.step(
            "SEM_FILTER(2019 lacks pandemic or public-health risk factor)",
            input_rows=len(pandemic_2020),
        ) as step:
            new_adopter_parts = []
            for documents in iter_selected_texts(
                "SEC",
                pandemic_2020,
                path_column="text_2019",
                output_column="document_2019",
                batch_size=BATCH_SIZE,
            ):
                selected = documents.sem_filter(
                    "The 2019 10-K filing {document_2019} does not include a pandemic "
                    "or public-health risk factor for the reporting company."
                )
                new_adopter_parts.append(
                    selected.drop(columns=["document_2019"], errors="ignore")
                )
            new_adopters = (
                pd.concat(new_adopter_parts, ignore_index=True)
                if new_adopter_parts
                else pandemic_2020.iloc[0:0].copy()
            )
            step.set_output(new_adopters)

        with tracker.step(
            "SEM_EXTRACT(annual risk-factor counts and 2020 industry sector)",
            input_rows=len(new_adopters),
        ) as step:
            extracted_parts = []
            for documents_2019 in iter_selected_texts(
                "SEC",
                new_adopters,
                path_column="text_2019",
                output_column="document_2019",
                batch_size=BATCH_SIZE,
            ):
                documents = load_selected_texts(
                    "SEC",
                    documents_2019,
                    path_column="text_2020",
                    output_column="document_2020",
                )
                batch = documents.sem_extract(
                    input_cols=["document_2019", "document_2020"],
                    output_cols={
                        "risk_factor_count_2019": (
                            "the total number of distinct risk factors in the 2019 "
                            "filing's Risk Factors section as an integer, or null "
                            "when the total cannot be determined"
                        ),
                        "risk_factor_count_2020": (
                            "the total number of distinct risk factors in the 2020 "
                            "filing's Risk Factors section as an integer, or null "
                            "when the total cannot be determined"
                        ),
                        "industry_sector_2020": (
                            "the reporting company's concise industry sector in the "
                            "2020 filing"
                        ),
                    },
                )
                for column in (
                    "risk_factor_count_2019",
                    "risk_factor_count_2020",
                ):
                    batch[column] = pd.to_numeric(
                        batch[column].map(parse_number), errors="coerce"
                    )
                batch["industry_sector_2020"] = batch[
                    "industry_sector_2020"
                ].map(clean_text)
                extracted_parts.append(
                    batch.drop(
                        columns=["document_2019", "document_2020"],
                        errors="ignore",
                    )
                )
            adopters = (
                pd.concat(extracted_parts, ignore_index=True)
                if extracted_parts
                else new_adopters.iloc[0:0].assign(
                    risk_factor_count_2019=pd.Series(dtype="float64"),
                    risk_factor_count_2020=pd.Series(dtype="float64"),
                    industry_sector_2020=pd.Series(dtype="object"),
                )
            )
            step.set_output(adopters)

        sector_counts = (
            adopters.groupby("industry_sector_2020", sort=False)["cik"]
            .nunique()
            .rename("adopter_count")
            .reset_index()
        )
        tracker.record(
            "GROUP_BY(industry_sector_2020, COUNT(DISTINCT cik) AS adopter_count)",
            len(adopters),
            len(sector_counts),
            output=sector_counts,
        )

        top_sectors = sector_counts.sort_values(
            ["adopter_count", "industry_sector_2020"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10).reset_index(drop=True)
        tracker.record(
            "ORDER_BY(adopter_count DESC, industry_sector_2020 ASC) -> LIMIT(10)",
            len(sector_counts),
            len(top_sectors),
            output=top_sectors,
        )

        mean_change = _mean_change(adopters)
        change_summary = {"mean_risk_factor_count_change": mean_change}
        tracker.record(
            "GROUP_BY([], ROUND(AVG(risk_factor_count_2020 - risk_factor_count_2019 IGNORE NULLS), 1))",
            len(adopters),
            1,
            output=change_summary,
        )

        answer = {
            "top_sectors": df_records(top_sectors),
            "mean_risk_factor_count_change": mean_change,
        }
        tracker.record(
            "PROJECT(top_sectors, mean_risk_factor_count_change)",
            {"top_sectors": len(top_sectors), "change_summary": 1},
            1,
            output=answer,
        )

    print(f"Result: {len(top_sectors)} sectors")
    save_output(TASK_ID, answer, elapsed=timer.elapsed, tracker=tracker)


if __name__ == "__main__":
    main()
