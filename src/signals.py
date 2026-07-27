"""Robot score, signal and ranking logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import StrategyConfig


def build_market_regime(market_features: pd.DataFrame) -> pd.Series:
    """Return XU100 Close > EMA200 as a dated Boolean series."""
    required = {"Date", "Close", "EMA200"}
    missing = required.difference(market_features.columns)
    if missing:
        raise KeyError(f"Piyasa filtresi için eksik sütunlar: {sorted(missing)}")

    market = (
        market_features.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .set_index("Date")
    )
    return (market["Close"] > market["EMA200"]).rename("MarketPositive")


def add_robot_scores(
    stock_features: pd.DataFrame,
    market_regime: pd.Series,
    config: StrategyConfig | None = None,
    include_reasons: bool = True,
) -> pd.DataFrame:
    """Vectorized implementation of Robot.ipynb's score_stock function."""
    config = config or StrategyConfig()
    result = stock_features.copy()

    market_regime = market_regime.sort_index()
    target_dates = pd.DatetimeIndex(result["Date"])
    result["MarketPositive"] = (
        market_regime.reindex(target_dates, method="ffill")
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    result["COND_MARKET"] = result["MarketPositive"]
    result["COND_CLOSE_EMA200"] = result["Close"] > result["EMA200"]
    result["COND_EMA50_EMA200"] = result["EMA50"] > result["EMA200"]
    result["COND_CLOSE_EMA20"] = result["Close"] > result["EMA20"]
    result["COND_RSI"] = result["RSI"] > config.minimum_rsi
    result["COND_MACD"] = result["MACD_HIST"] > 0
    result["COND_ADX"] = result["ADX"] > config.minimum_adx
    result["COND_VOLUME"] = (
        result["Volume"] > result["VOL_SMA20"] * config.volume_multiplier
    )
    result["COND_BREAKOUT"] = result["Close"] > result["HIGH_20_PREV"]

    result["Score"] = (
        result["COND_MARKET"].astype(int) * config.market_score
        + result["COND_CLOSE_EMA200"].astype(int)
        * config.close_above_ema200_score
        + result["COND_EMA50_EMA200"].astype(int)
        * config.ema50_above_ema200_score
        + result["COND_CLOSE_EMA20"].astype(int)
        * config.close_above_ema20_score
        + result["COND_RSI"].astype(int) * config.rsi_score
        + result["COND_MACD"].astype(int) * config.macd_score
        + result["COND_ADX"].astype(int) * config.adx_score
        + result["COND_VOLUME"].astype(int) * config.volume_score
        + result["COND_BREAKOUT"].astype(int) * config.breakout_score
    )

    result["Signal"] = np.select(
        [
            result["Score"] >= config.buy_score,
            result["Score"] >= config.watch_score,
        ],
        ["AL", "İZLE"],
        default="ALMA",
    )

    if include_reasons:
        reason_pairs = [
            ("COND_MARKET", "Piyasa pozitif"),
            ("COND_CLOSE_EMA200", "EMA200 üstü"),
            ("COND_EMA50_EMA200", "EMA50 > EMA200"),
            ("COND_CLOSE_EMA20", "EMA20 üstü"),
            ("COND_RSI", "RSI güçlü"),
            ("COND_MACD", "MACD pozitif"),
            ("COND_ADX", "ADX trend var"),
            ("COND_VOLUME", "Hacim güçlü"),
            ("COND_BREAKOUT", "20 gün zirve kırılımı"),
        ]

        def make_reasons(row: pd.Series) -> str:
            return ", ".join(
                label for column, label in reason_pairs if bool(row[column])
            )

        result["Reasons"] = result.apply(make_reasons, axis=1)

    result["ReferenceStop"] = (
        result["Close"] - config.initial_stop_atr * result["ATR"]
    )

    return result


def rank_candidates(daily_rows: pd.DataFrame) -> pd.DataFrame:
    """Rank Robot AL candidates by Score, RET_126, RET_63 and ADX."""
    candidates = daily_rows.loc[daily_rows["Signal"].eq("AL")].copy()
    return candidates.sort_values(
        ["Score", "RET_126", "RET_63", "ADX"],
        ascending=False,
    ).reset_index(drop=True)
