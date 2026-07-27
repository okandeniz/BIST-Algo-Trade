"""Robot-compatible data cleaning, split-gap repair and quality checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DataConfig


REQUIRED_COLUMNS = {
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
    "Dividends", "Stock Splits",
}
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


@dataclass
class QualityPipelineResult:
    enriched: pd.DataFrame
    clean: pd.DataFrame
    summary: pd.DataFrame
    split_repairs: pd.DataFrame


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise KeyError(f"Veri kalite kontrolü için eksik sütunlar: {sorted(missing)}")


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Robot notebook's clean_yf_data logic without silently deleting rows."""
    validate_schema(df)
    result = df.copy()

    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    numeric_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "Dividends", "Stock Splits",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return (
        result.drop_duplicates(["Ticker", "Date"], keep="last")
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )


def add_quality_flags(
    df: pd.DataFrame,
    gap_threshold: float = 0.35,
    large_return_threshold: float = 0.35,
) -> pd.DataFrame:
    """Add quality flags without changing prices."""
    result = standardize_types(df)

    result["MissingDate"] = result["Date"].isna()
    result["MissingOHLCV"] = result[
        ["Open", "High", "Low", "Close", "Volume"]
    ].isna().any(axis=1)

    result["NonPositivePrice"] = result[PRICE_COLUMNS].le(0).any(axis=1)
    result["NegativeVolume"] = result["Volume"].lt(0)
    result["ZeroVolume"] = result["Volume"].fillna(0).eq(0)

    scale = (
        result[PRICE_COLUMNS].abs().max(axis=1).clip(lower=1.0)
    )
    tolerance = scale * 1e-6

    expected_high = result[["Open", "Low", "Close"]].max(axis=1)
    expected_low = result[["Open", "High", "Close"]].min(axis=1)

    result["InvalidOHLC"] = (
        (result["High"] + tolerance < expected_high)
        | (result["Low"] - tolerance > expected_low)
    )

    result["FlatOHLC"] = (
        (result["Open"] - result["High"]).abs().le(tolerance)
        & (result["Open"] - result["Low"]).abs().le(tolerance)
        & (result["Open"] - result["Close"]).abs().le(tolerance)
    )

    result["HasDividend"] = result["Dividends"].fillna(0).ne(0)
    result["HasSplit"] = result["Stock Splits"].fillna(0).ne(0)
    result["HasCorporateAction"] = result["HasDividend"] | result["HasSplit"]

    result["SyntheticNoTradeBar"] = (
        result["ZeroVolume"]
        & result["FlatOHLC"]
        & ~result["HasCorporateAction"]
    )

    grouped = result.groupby("Ticker", sort=False)
    previous_close = grouped["Close"].shift(1)

    result["GapRatio"] = result["Open"] / previous_close
    result["CloseReturn"] = grouped["Close"].pct_change(fill_method=None)

    result["SuspectedSplitGap"] = (
        result["GapRatio"].gt(0)
        & result["GapRatio"].lt(gap_threshold)
    )
    result["LargeCloseReturn"] = (
        result["CloseReturn"].abs().gt(large_return_threshold)
    )

    result["ValidPriceBar"] = ~(
        result["MissingDate"]
        | result["MissingOHLCV"]
        | result["NonPositivePrice"]
        | result["NegativeVolume"]
        | result["InvalidOHLC"]
    )
    result["TradableBar"] = result["ValidPriceBar"] & result["Volume"].gt(0)

    return result


def repair_split_gaps(
    df: pd.DataFrame,
    threshold: float = 0.35,
    ratio_tolerance: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the Robot notebook's historical split repair with safety checks.

    Only integer-like downward gaps are repaired. Every repair is logged.
    """
    result = standardize_types(df)
    repair_records: list[dict[str, object]] = []

    repaired_frames: list[pd.DataFrame] = []

    for ticker, ticker_df in result.groupby("Ticker", sort=False):
        work = ticker_df.sort_values("Date").copy().reset_index(drop=True)
        work["ManualSplitRepair"] = False
        work["ManualSplitRatio"] = np.nan

        # Candidate dates are recalculated from the current work table.
        candidate_positions = list(
            np.flatnonzero(
                (work["Open"] / work["Close"].shift(1)).lt(threshold).fillna(False)
            )
        )

        for position in candidate_positions:
            if position == 0:
                continue

            split_date = work.loc[position, "Date"]
            previous_close = float(work.loc[position - 1, "Close"])
            current_open = float(work.loc[position, "Open"])

            if previous_close <= 0 or current_open <= 0:
                continue

            raw_factor = current_open / previous_close
            split_ratio = int(round(1 / raw_factor))

            if split_ratio <= 1:
                continue

            expected_factor = 1 / split_ratio
            relative_error = abs(raw_factor - expected_factor) / expected_factor

            if relative_error > ratio_tolerance:
                continue

            factor = expected_factor
            historical_mask = work.index < position

            work.loc[historical_mask, PRICE_COLUMNS] = (
                work.loc[historical_mask, PRICE_COLUMNS] * factor
            )
            work.loc[historical_mask, "Volume"] = (
                work.loc[historical_mask, "Volume"] / factor
            )

            work.loc[position, "ManualSplitRepair"] = True
            work.loc[position, "ManualSplitRatio"] = split_ratio

            repair_records.append({
                "Ticker": ticker,
                "Date": split_date,
                "RawFactor": raw_factor,
                "EstimatedSplitRatio": split_ratio,
                "AppliedFactor": factor,
                "RelativeError": relative_error,
                "YahooSplitValue": work.loc[position, "Stock Splits"],
            })

        repaired_frames.append(work)

    repaired = (
        pd.concat(repaired_frames, ignore_index=True)
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )
    repairs = pd.DataFrame(repair_records)
    return repaired, repairs


def create_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    required_flags = {
        "ValidPriceBar", "TradableBar", "MissingOHLCV",
        "NonPositivePrice", "InvalidOHLC", "ZeroVolume",
        "SyntheticNoTradeBar", "HasDividend", "HasSplit",
        "SuspectedSplitGap", "LargeCloseReturn",
    }
    missing = required_flags.difference(df.columns)
    if missing:
        raise KeyError(
            "Önce add_quality_flags() çalıştırılmalıdır. "
            f"Eksik sütunlar: {sorted(missing)}"
        )

    summary = (
        df.groupby("Ticker")
        .agg(
            StartDate=("Date", "min"),
            EndDate=("Date", "max"),
            RowCount=("Date", "size"),
            ValidPriceBars=("ValidPriceBar", "sum"),
            TradableBars=("TradableBar", "sum"),
            MissingRows=("MissingOHLCV", "sum"),
            NonPositivePriceRows=("NonPositivePrice", "sum"),
            InvalidOHLCRows=("InvalidOHLC", "sum"),
            ZeroVolumeRows=("ZeroVolume", "sum"),
            SyntheticNoTradeBars=("SyntheticNoTradeBar", "sum"),
            DividendEvents=("HasDividend", "sum"),
            SplitEvents=("HasSplit", "sum"),
            SuspectedSplitGaps=("SuspectedSplitGap", "sum"),
            LargeCloseReturns=("LargeCloseReturn", "sum"),
        )
        .reset_index()
    )

    summary["ValidBarRatio"] = summary["ValidPriceBars"] / summary["RowCount"]
    summary["TradableBarRatio"] = summary["TradableBars"] / summary["RowCount"]

    return summary.sort_values(
        ["SuspectedSplitGaps", "InvalidOHLCRows", "ZeroVolumeRows"],
        ascending=False,
    ).reset_index(drop=True)


def run_quality_pipeline(
    df: pd.DataFrame,
    config: DataConfig | None = None,
    apply_split_repairs: bool = True,
) -> QualityPipelineResult:
    """Flag raw data, optionally repair unresolved split gaps, then clean."""
    config = config or DataConfig()

    if apply_split_repairs:
        repaired, repair_log = repair_split_gaps(
            df,
            threshold=config.split_gap_threshold,
            ratio_tolerance=config.split_ratio_tolerance,
        )
    else:
        repaired = standardize_types(df)
        repaired["ManualSplitRepair"] = False
        repaired["ManualSplitRatio"] = np.nan
        repair_log = pd.DataFrame()

    enriched = add_quality_flags(
        repaired,
        gap_threshold=config.split_gap_threshold,
    )

    clean = (
        enriched.loc[enriched["TradableBar"]]
        .copy()
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )
    summary = create_quality_summary(enriched)

    return QualityPipelineResult(
        enriched=enriched,
        clean=clean,
        summary=summary,
        split_repairs=repair_log,
    )
