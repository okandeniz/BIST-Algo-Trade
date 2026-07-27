"""Leakage-safe meta-label dataset construction for the Robot strategy.

One row represents one primary-strategy entry event. Entry and exit timing
follow the final Robot rules:

- Signal generated at the completed daily close
- Entry at the next available open
- Intraday initial stop
- Trailing and LOW10 close rules executed at the next open
- Commissions and slippage included in the target return

Events do not overlap for the same ticker. This reduces repeated,
near-identical samples while preserving the primary strategy's state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import json
import numpy as np
import pandas as pd

from src.config import PortfolioConfig, StrategyConfig


BASE_FEATURE_COLUMNS = [
    "Score",
    "RSI",
    "ADX",
    "ATR_PCT",
    "MACD_HIST_PCT",
    "VOLUME_RATIO",
    "RET_5",
    "RET_20",
    "RET_63",
    "RET_126",
    "VOLATILITY_20",
    "CLOSE_VS_EMA20",
    "CLOSE_VS_EMA50",
    "CLOSE_VS_EMA200",
    "EMA50_VS_EMA200",
    "EMA20_SLOPE_5",
    "EMA50_SLOPE_5",
    "ADX_CHANGE_5",
    "RSI_CHANGE_5",
    "BREAKOUT_DISTANCE",
    "MARKET_DISTANCE_EMA200",
    "MARKET_RET_20",
    "MARKET_RET_63",
    "MARKET_VOLATILITY_20",
    "SCORE_RANK_PCT",
    "ADX_RANK_PCT",
    "RET_63_RANK_PCT",
    "RET_126_RANK_PCT",
    "VOLUME_RATIO_RANK_PCT",
    "Month",
    "DayOfWeek",
]


TARGET_COLUMNS = [
    "Meta_Label",
    "Meta_Label_1R",
    "Outperform_BIST100_Label",
    "Net_Return",
    "Net_Return_%",
    "Initial_Risk_%",
    "R_Multiple",
    "MFE_%",
    "MAE_%",
    "Market_Holding_Return_%",
    "Excess_Return_vs_BIST100_%",
    "Holding_Bars",
    "Holding_Calendar_Days",
    "Exit_Reason",
    "Is_Censored",
    "Is_Outlier",
]


@dataclass(frozen=True)
class EventOutcome:
    exit_position: int
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str
    net_return: float
    initial_risk_pct: float
    r_multiple: float
    mfe_pct: float
    mae_pct: float
    holding_bars: int
    holding_calendar_days: int
    is_censored: bool
    is_outlier: bool


def add_meta_features(
    scored_prices: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame:
    """Add features calculated only from current and historical bars."""
    required_stock = {
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "MACD_HIST",
        "ADX",
        "ATR",
        "VOL_SMA20",
        "HIGH_20_PREV",
        "LOW_10_PREV",
        "RET_63",
        "RET_126",
        "Score",
        "Signal",
        "MarketPositive",
    }
    missing_stock = required_stock.difference(scored_prices.columns)

    if missing_stock:
        raise KeyError(
            "ML özellikleri için eksik hisse sütunları: "
            f"{sorted(missing_stock)}"
        )

    required_market = {
        "Date",
        "Close",
        "EMA200",
    }
    missing_market = required_market.difference(
        market_features.columns
    )

    if missing_market:
        raise KeyError(
            "ML özellikleri için eksik piyasa sütunları: "
            f"{sorted(missing_market)}"
        )

    data = scored_prices.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    grouped = data.groupby("Ticker", sort=False)

    data["RET_5"] = grouped["Close"].pct_change(
        5,
        fill_method=None,
    )
    data["RET_20"] = grouped["Close"].pct_change(
        20,
        fill_method=None,
    )

    daily_return = grouped["Close"].pct_change(
        fill_method=None
    )

    data["VOLATILITY_20"] = (
        daily_return.groupby(data["Ticker"])
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(252)
    )

    data["ATR_PCT"] = data["ATR"] / data["Close"]
    data["MACD_HIST_PCT"] = (
        data["MACD_HIST"] / data["Close"]
    )
    data["VOLUME_RATIO"] = (
        data["Volume"] / data["VOL_SMA20"]
    )

    data["CLOSE_VS_EMA20"] = (
        data["Close"] / data["EMA20"] - 1
    )
    data["CLOSE_VS_EMA50"] = (
        data["Close"] / data["EMA50"] - 1
    )
    data["CLOSE_VS_EMA200"] = (
        data["Close"] / data["EMA200"] - 1
    )
    data["EMA50_VS_EMA200"] = (
        data["EMA50"] / data["EMA200"] - 1
    )

    data["EMA20_SLOPE_5"] = grouped["EMA20"].pct_change(
        5,
        fill_method=None,
    )
    data["EMA50_SLOPE_5"] = grouped["EMA50"].pct_change(
        5,
        fill_method=None,
    )
    data["ADX_CHANGE_5"] = grouped["ADX"].diff(5)
    data["RSI_CHANGE_5"] = grouped["RSI"].diff(5)

    data["BREAKOUT_DISTANCE"] = (
        data["Close"] / data["HIGH_20_PREV"] - 1
    )

    market = (
        market_features.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .copy()
    )
    market["Date"] = pd.to_datetime(market["Date"])
    market["MARKET_DISTANCE_EMA200"] = (
        market["Close"] / market["EMA200"] - 1
    )
    market["MARKET_RET_20"] = market["Close"].pct_change(
        20,
        fill_method=None,
    )
    market["MARKET_RET_63"] = market["Close"].pct_change(
        63,
        fill_method=None,
    )
    market_daily_return = market["Close"].pct_change(
        fill_method=None
    )
    market["MARKET_VOLATILITY_20"] = (
        market_daily_return.rolling(20).std()
        * np.sqrt(252)
    )

    data = data.merge(
        market[
            [
                "Date",
                "MARKET_DISTANCE_EMA200",
                "MARKET_RET_20",
                "MARKET_RET_63",
                "MARKET_VOLATILITY_20",
            ]
        ],
        on="Date",
        how="left",
        validate="many_to_one",
    )

    rank_specs = {
        "SCORE_RANK_PCT": "Score",
        "ADX_RANK_PCT": "ADX",
        "RET_63_RANK_PCT": "RET_63",
        "RET_126_RANK_PCT": "RET_126",
        "VOLUME_RATIO_RANK_PCT": "VOLUME_RATIO",
    }

    for output_column, source_column in rank_specs.items():
        data[output_column] = (
            data.groupby("Date")[source_column]
            .rank(
                pct=True,
                method="average",
            )
        )

    data["Month"] = data["Date"].dt.month.astype("int8")
    data["DayOfWeek"] = (
        data["Date"].dt.dayofweek.astype("int8")
    )

    return data


def _net_return(
    entry_price: float,
    exit_price: float,
    portfolio_config: PortfolioConfig,
) -> float:
    return (
        exit_price
        * (1 - portfolio_config.commission_rate)
        / (
            entry_price
            * (1 + portfolio_config.commission_rate)
        )
        - 1
    )


def _initial_risk_pct(
    entry_price: float,
    stop_loss: float,
    portfolio_config: PortfolioConfig,
) -> float:
    entry_cost = (
        entry_price
        * (1 + portfolio_config.commission_rate)
    )
    estimated_stop_proceeds = (
        stop_loss
        * (1 - portfolio_config.slippage_rate)
        * (1 - portfolio_config.commission_rate)
    )
    return (
        entry_cost - estimated_stop_proceeds
    ) / entry_cost


def _simulate_event(
    ticker_data: pd.DataFrame,
    signal_position: int,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
) -> EventOutcome | None:
    """Simulate one primary-strategy trade from one close signal."""
    entry_position = signal_position + 1

    if entry_position >= len(ticker_data):
        return None

    signal_row = ticker_data.iloc[signal_position]
    entry_row = ticker_data.iloc[entry_position]

    entry_price = float(entry_row["Open"]) * (
        1 + portfolio_config.slippage_rate
    )
    signal_atr = float(signal_row["ATR"])

    if (
        not np.isfinite(entry_price)
        or not np.isfinite(signal_atr)
        or entry_price <= 0
        or signal_atr <= 0
    ):
        return None

    stop_loss = (
        entry_price
        - strategy_config.initial_stop_atr * signal_atr
    )

    if stop_loss <= 0:
        return None

    highest_price = entry_price
    maximum_favorable_price = entry_price
    minimum_adverse_price = entry_price

    exit_position = len(ticker_data) - 1
    exit_date = pd.Timestamp(
        ticker_data.iloc[exit_position]["Date"]
    )
    exit_price = float(
        ticker_data.iloc[exit_position]["Close"]
    ) * (1 - portfolio_config.slippage_rate)
    exit_reason = "End of Data"
    is_censored = True

    for current_position in range(
        entry_position,
        len(ticker_data),
    ):
        row = ticker_data.iloc[current_position]

        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        atr = float(row["ATR"])
        low_10_prev = float(row["LOW_10_PREV"])

        maximum_favorable_price = max(
            maximum_favorable_price,
            high,
        )
        minimum_adverse_price = min(
            minimum_adverse_price,
            low,
        )
        highest_price = max(highest_price, high)

        if open_price <= stop_loss:
            exit_position = current_position
            exit_date = pd.Timestamp(row["Date"])
            exit_price = open_price * (
                1 - portfolio_config.slippage_rate
            )
            exit_reason = "Stop Loss Gap"
            is_censored = False
            break

        if low <= stop_loss:
            exit_position = current_position
            exit_date = pd.Timestamp(row["Date"])
            exit_price = stop_loss * (
                1 - portfolio_config.slippage_rate
            )
            exit_reason = "Stop Loss"
            is_censored = False
            break

        trailing_level = (
            highest_price
            - strategy_config.trailing_stop_atr * atr
        )
        profit_pct = (
            close - entry_price
        ) / entry_price

        close_exit_reason: str | None = None

        if (
            profit_pct
            > strategy_config.trailing_activation_return
            and close < trailing_level
        ):
            close_exit_reason = "Trailing Stop"
        elif close < low_10_prev:
            close_exit_reason = "LOW10 Altı"

        if close_exit_reason is not None:
            next_position = current_position + 1

            if next_position < len(ticker_data):
                next_row = ticker_data.iloc[next_position]
                exit_position = next_position
                exit_date = pd.Timestamp(
                    next_row["Date"]
                )
                exit_price = float(
                    next_row["Open"]
                ) * (
                    1 - portfolio_config.slippage_rate
                )
                exit_reason = close_exit_reason
                is_censored = False
            else:
                exit_position = current_position
                exit_date = pd.Timestamp(row["Date"])
                exit_price = close * (
                    1 - portfolio_config.slippage_rate
                )
                exit_reason = "End of Data"
                is_censored = True

            break

    net_return = _net_return(
        entry_price,
        exit_price,
        portfolio_config,
    )
    initial_risk_pct = _initial_risk_pct(
        entry_price,
        stop_loss,
        portfolio_config,
    )
    r_multiple = (
        net_return / initial_risk_pct
        if initial_risk_pct > 0
        else np.nan
    )

    entry_date = pd.Timestamp(entry_row["Date"])

    return EventOutcome(
        exit_position=exit_position,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
        net_return=net_return,
        initial_risk_pct=initial_risk_pct,
        r_multiple=r_multiple,
        mfe_pct=(
            maximum_favorable_price / entry_price - 1
        ) * 100,
        mae_pct=(
            minimum_adverse_price / entry_price - 1
        ) * 100,
        holding_bars=(
            exit_position - entry_position + 1
        ),
        holding_calendar_days=(
            exit_date - entry_date
        ).days,
        is_censored=is_censored,
        is_outlier=not (
            portfolio_config.minimum_valid_return
            <= net_return
            <= portfolio_config.maximum_valid_return
        ),
    )


def _market_return_lookup(
    market_features: pd.DataFrame,
) -> pd.Series:
    market = (
        market_features.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .set_index("Date")
    )
    market.index = pd.to_datetime(market.index)
    return market["Close"].astype(float)


def _period_label(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp("2022-12-31"):
        return "Development"
    if date <= pd.Timestamp("2024-12-31"):
        return "Validation"
    return "Audit_2025_Plus"


def build_meta_label_dataset(
    featured_prices: pd.DataFrame,
    market_features: pd.DataFrame,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build one non-overlapping event row per ticker entry signal."""
    features = list(
        feature_columns or BASE_FEATURE_COLUMNS
    )

    required = {
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "LOW_10_PREV",
        "Score",
        "Signal",
    }.union(features)

    missing = required.difference(featured_prices.columns)

    if missing:
        raise KeyError(
            "Meta-label dataset için eksik sütunlar: "
            f"{sorted(missing)}"
        )

    market_close = _market_return_lookup(
        market_features
    )
    records: list[dict[str, Any]] = []

    for ticker, ticker_df in featured_prices.groupby(
        "Ticker",
        sort=False,
    ):
        work = (
            ticker_df.sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )

        position = 0

        while position < len(work) - 1:
            signal_row = work.iloc[position]

            if signal_row["Signal"] != "AL":
                position += 1
                continue

            if signal_row[features].isna().any():
                position += 1
                continue

            outcome = _simulate_event(
                ticker_data=work,
                signal_position=position,
                strategy_config=strategy_config,
                portfolio_config=portfolio_config,
            )

            if outcome is None:
                position += 1
                continue

            signal_date = pd.Timestamp(
                signal_row["Date"]
            )
            entry_date = pd.Timestamp(
                work.iloc[position + 1]["Date"]
            )

            market_start = market_close.get(signal_date)
            market_end = market_close.get(outcome.exit_date)

            market_return = (
                float(market_end / market_start - 1)
                if (
                    market_start is not None
                    and market_end is not None
                    and market_start > 0
                )
                else np.nan
            )

            record: dict[str, Any] = {
                "Ticker": ticker,
                "Signal_Date": signal_date,
                "Entry_Date": entry_date,
                "Exit_Date": outcome.exit_date,
                "Signal_Close": float(
                    signal_row["Close"]
                ),
                "Entry_Open_Raw": float(
                    work.iloc[position + 1]["Open"]
                ),
                "Entry_Price": float(
                    work.iloc[position + 1]["Open"]
                ) * (
                    1
                    + portfolio_config.slippage_rate
                ),
                "Exit_Price": outcome.exit_price,
                "Period": _period_label(signal_date),
                "Meta_Label": int(
                    outcome.net_return > 0
                ),
                "Meta_Label_1R": int(
                    outcome.r_multiple >= 1
                )
                if np.isfinite(outcome.r_multiple)
                else 0,
                "Outperform_BIST100_Label": int(
                    outcome.net_return > market_return
                )
                if np.isfinite(market_return)
                else np.nan,
                "Net_Return": outcome.net_return,
                "Net_Return_%": (
                    outcome.net_return * 100
                ),
                "Initial_Risk_%": (
                    outcome.initial_risk_pct * 100
                ),
                "R_Multiple": outcome.r_multiple,
                "MFE_%": outcome.mfe_pct,
                "MAE_%": outcome.mae_pct,
                "Market_Holding_Return_%": (
                    market_return * 100
                )
                if np.isfinite(market_return)
                else np.nan,
                "Excess_Return_vs_BIST100_%": (
                    (
                        outcome.net_return
                        - market_return
                    )
                    * 100
                )
                if np.isfinite(market_return)
                else np.nan,
                "Holding_Bars": outcome.holding_bars,
                "Holding_Calendar_Days": (
                    outcome.holding_calendar_days
                ),
                "Exit_Reason": outcome.exit_reason,
                "Is_Censored": outcome.is_censored,
                "Is_Outlier": outcome.is_outlier,
            }

            for feature in features:
                value = signal_row[feature]
                record[feature] = (
                    float(value)
                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.integer,
                            np.floating,
                        ),
                    )
                    else value
                )

            records.append(record)

            # The same ticker cannot produce another event while
            # this simulated primary-strategy trade is open.
            position = max(
                position + 1,
                outcome.exit_position,
            )

    dataset = pd.DataFrame(records)

    if dataset.empty:
        return dataset

    return (
        dataset.sort_values(
            ["Signal_Date", "Ticker"]
        )
        .reset_index(drop=True)
    )


def leakage_audit(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Run structural checks before any model is trained."""
    if dataset.empty:
        return pd.DataFrame(
            [
                {
                    "Check": "Dataset non-empty",
                    "Passed": False,
                    "Details": "Dataset boş.",
                }
            ]
        )

    ordered = dataset.sort_values(
        ["Ticker", "Signal_Date"]
    ).copy()

    next_signal = ordered.groupby("Ticker")[
        "Signal_Date"
    ].shift(-1)

    overlap_mask = (
        next_signal.notna()
        & (next_signal < ordered["Exit_Date"])
    )

    checks = [
        {
            "Check": "Signal before entry",
            "Passed": bool(
                (
                    dataset["Signal_Date"]
                    < dataset["Entry_Date"]
                ).all()
            ),
            "Details": (
                "Her giriş sinyal gününden sonra olmalı."
            ),
        },
        {
            "Check": "Entry not after exit",
            "Passed": bool(
                (
                    dataset["Entry_Date"]
                    <= dataset["Exit_Date"]
                ).all()
            ),
            "Details": (
                "Giriş tarihi çıkış tarihinden sonra olamaz."
            ),
        },
        {
            "Check": "Unique ticker-signal events",
            "Passed": not bool(
                dataset.duplicated(
                    ["Ticker", "Signal_Date"]
                ).any()
            ),
            "Details": (
                "Ticker ve sinyal tarihi benzersiz olmalı."
            ),
        },
        {
            "Check": "No same-ticker overlapping events",
            "Passed": not bool(overlap_mask.any()),
            "Details": (
                f"Çakışan olay sayısı: {int(overlap_mask.sum())}"
            ),
        },
        {
            "Check": "Binary primary label",
            "Passed": bool(
                set(
                    dataset["Meta_Label"]
                    .dropna()
                    .unique()
                ).issubset({0, 1})
            ),
            "Details": "Meta_Label yalnızca 0/1 olmalı.",
        },
        {
            "Check": "No missing base features",
            "Passed": not bool(
                dataset[
                    BASE_FEATURE_COLUMNS
                ].isna().any().any()
            ),
            "Details": (
                "Model özelliklerinde eksik değer olmamalı."
            ),
        },
    ]

    return pd.DataFrame(checks)


def save_feature_manifest(
    feature_columns: Iterable[str],
    path: str | Path,
) -> Path:
    """Save the exact model feature list for reproducibility."""
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "feature_columns": list(feature_columns),
                "target_column": "Meta_Label",
                "secondary_targets": [
                    "Meta_Label_1R",
                    "Outperform_BIST100_Label",
                ],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path
