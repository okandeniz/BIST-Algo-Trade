"""Technical indicators used by the Robot notebook."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange


REQUIRED_COLUMNS = {
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
}


def add_indicators_single(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Robot.ipynb's indicator set for one symbol."""
    result = df.sort_values("Date").copy().reset_index(drop=True)

    result["EMA20"] = EMAIndicator(result["Close"], window=20).ema_indicator()
    result["EMA50"] = EMAIndicator(result["Close"], window=50).ema_indicator()
    result["EMA200"] = EMAIndicator(result["Close"], window=200).ema_indicator()

    result["RSI"] = RSIIndicator(result["Close"], window=14).rsi()

    macd = MACD(
        close=result["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )
    result["MACD_HIST"] = macd.macd_diff()

    adx = ADXIndicator(
        high=result["High"],
        low=result["Low"],
        close=result["Close"],
        window=14,
    )
    result["ADX"] = adx.adx()

    atr = AverageTrueRange(
        high=result["High"],
        low=result["Low"],
        close=result["Close"],
        window=14,
    )
    result["ATR"] = atr.average_true_range()

    result["VOL_SMA20"] = result["Volume"].rolling(20).mean()
    result["HIGH_20_PREV"] = result["Close"].rolling(20).max().shift(1)
    result["LOW_10_PREV"] = result["Low"].rolling(10).min().shift(1)
    result["RET_63"] = result["Close"].pct_change(63, fill_method=None)
    result["RET_126"] = result["Close"].pct_change(126, fill_method=None)

    required_features = [
        "EMA20", "EMA50", "EMA200", "RSI", "MACD_HIST", "ADX",
        "ATR", "VOL_SMA20", "HIGH_20_PREV", "LOW_10_PREV",
        "RET_63", "RET_126",
    ]

    return result.dropna(subset=required_features).reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators independently for each ticker."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise KeyError(f"İndikatörler için eksik sütunlar: {sorted(missing)}")

    frames = []
    for _, ticker_df in df.groupby("Ticker", sort=False):
        if len(ticker_df) < 200:
            continue
        frames.append(add_indicators_single(ticker_df))

    if not frames:
        return pd.DataFrame(columns=list(df.columns))

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )
