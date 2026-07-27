"""BIST ticker loading and Yahoo Finance download utilities.

The Robot notebook works with auto-adjusted OHLC prices. This module keeps
that convention: Open, High, Low and Close are the canonical adjusted prices.
Corporate-action columns are retained for audit and quality control.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm

from src.config import DataConfig


STANDARD_COLUMNS = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Capital Gains",
    "Repaired?",
]


def load_bist_tickers(
    excel_path: str | Path,
    ticker_column: str = "Kod",
) -> list[str]:
    """Read BIST ticker codes and normalize them to Yahoo's `.IS` format."""
    path = Path(excel_path)

    if not path.exists():
        raise FileNotFoundError(f"Excel dosyası bulunamadı: {path.resolve()}")

    df = pd.read_excel(path)

    if ticker_column not in df.columns:
        raise KeyError(
            f"'{ticker_column}' sütunu bulunamadı. "
            f"Mevcut sütunlar: {df.columns.tolist()}"
        )

    tickers = (
        df[ticker_column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    tickers = tickers.map(
        lambda value: value if value.endswith(".IS") else f"{value}.IS"
    )

    return tickers.drop_duplicates().tolist()


def _normalize_history(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert a yfinance history result into the project's long format."""
    if df.empty:
        raise ValueError(f"{ticker} için boş veri döndü.")

    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    result = result.loc[:, ~result.columns.duplicated()]

    result.index.name = "Date"
    result = result.reset_index()

    dates = pd.to_datetime(result["Date"], errors="coerce")
    if dates.dt.tz is not None:
        # Preserve the exchange-local calendar date.
        dates = dates.dt.tz_localize(None)
    result["Date"] = dates.dt.normalize()

    defaults = {
        "Dividends": 0.0,
        "Stock Splits": 0.0,
        "Capital Gains": 0.0,
        "Repaired?": False,
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"{ticker} eksik sütunlar: {sorted(missing)}")

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Capital Gains",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["Repaired?"] = result["Repaired?"].fillna(False).astype(bool)
    result["Ticker"] = ticker

    return (
        result[STANDARD_COLUMNS]
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )


def download_symbol(
    ticker: str,
    config: DataConfig | None = None,
) -> pd.DataFrame:
    """Download one ticker using the Robot notebook's adjusted-price convention."""
    config = config or DataConfig()
    last_error: Exception | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            history = yf.Ticker(ticker).history(
                start=config.start,
                end=config.end,
                interval=config.interval,
                auto_adjust=config.auto_adjust,
                actions=True,
                repair=config.yfinance_repair,
                keepna=True,
                raise_errors=True,
            )
            return _normalize_history(history, ticker)

        except Exception as error:
            last_error = error
            if attempt < config.max_retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"{ticker} verisi {config.max_retries} denemede indirilemedi. "
        f"Son hata: {last_error}"
    )


def download_universe(
    tickers: Iterable[str],
    config: DataConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download multiple symbols and return combined prices plus an error report."""
    config = config or DataConfig()
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for ticker in tqdm(list(tickers), desc="Yahoo Finance verileri"):
        try:
            frames.append(download_symbol(ticker, config))
        except Exception as error:
            errors.append({"Ticker": ticker, "Error": str(error)})
        time.sleep(config.pause_seconds)

    if not frames:
        raise RuntimeError("Hiçbir sembol için veri indirilemedi.")

    prices = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )
    error_df = pd.DataFrame(errors, columns=["Ticker", "Error"])
    return prices, error_df


def download_robot_bundle(
    tickers: Iterable[str],
    config: DataConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download BIST shares and XU100 with the same date/adjustment settings."""
    config = config or DataConfig()
    stock_prices, errors = download_universe(tickers, config)
    market_prices = download_symbol(config.market_symbol, config)
    return stock_prices, market_prices, errors


def save_robot_bundle(
    stock_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    errors: pd.DataFrame,
    project_root: str | Path,
) -> dict[str, Path]:
    """Persist raw Robot-compatible data files."""
    project_root = Path(project_root)
    raw_dir = project_root / "data" / "raw"
    results_dir = project_root / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "stocks": raw_dir / "bist100_robot_raw.parquet",
        "market": raw_dir / "xu100_robot_raw.parquet",
        "errors": results_dir / "download_errors.csv",
    }

    stock_prices.to_parquet(paths["stocks"], index=False)
    market_prices.to_parquet(paths["market"], index=False)
    errors.to_csv(paths["errors"], index=False)
    return paths
