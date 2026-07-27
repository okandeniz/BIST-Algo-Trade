"""Daily Robot buy/sell plan generation for paper trading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import PortfolioConfig, StrategyConfig
from src.signals import rank_candidates


@dataclass
class DailyPlan:
    signal_date: pd.Timestamp
    summary: pd.DataFrame
    buy_orders: pd.DataFrame
    sell_orders: pd.DataFrame
    hold_positions: pd.DataFrame
    ranked_candidates: pd.DataFrame


def choose_signal_date(
    scored_prices: pd.DataFrame,
    minimum_coverage_ratio: float = 0.60,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Choose the latest date with sufficient cross-sectional coverage."""
    if not 0 < minimum_coverage_ratio <= 1:
        raise ValueError(
            "minimum_coverage_ratio 0 ile 1 arasında olmalıdır."
        )

    data = scored_prices.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    if as_of_date is not None:
        data = data.loc[
            data["Date"] <= pd.Timestamp(as_of_date)
        ]

    if data.empty:
        raise ValueError("Sinyal tarihi seçmek için veri bulunmuyor.")

    total_tickers = data["Ticker"].nunique()
    required_count = max(
        1,
        int(np.ceil(total_tickers * minimum_coverage_ratio)),
    )

    coverage = data.groupby("Date")["Ticker"].nunique()
    valid_dates = coverage.loc[
        coverage >= required_count
    ].index

    if len(valid_dates) == 0:
        raise ValueError(
            "Minimum kapsama oranını sağlayan tarih bulunamadı."
        )

    return pd.Timestamp(valid_dates.max())


def _latest_rows_on_date(
    scored_prices: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    result = scored_prices.loc[
        pd.to_datetime(scored_prices["Date"]).eq(signal_date)
    ].copy()

    return (
        result.sort_values(
            ["Score", "RET_126", "RET_63", "ADX"],
            ascending=False,
        )
        .drop_duplicates("Ticker", keep="first")
        .reset_index(drop=True)
    )


def _latest_price_map(
    latest_rows: pd.DataFrame,
) -> dict[str, float]:
    return dict(
        zip(
            latest_rows["Ticker"],
            latest_rows["Close"],
            strict=False,
        )
    )


def _portfolio_equity(
    state: dict[str, Any],
    latest_rows: pd.DataFrame,
) -> float:
    price_map = _latest_price_map(latest_rows)
    value = float(state["cash"])

    for ticker, position in state["positions"].items():
        price = price_map.get(
            ticker,
            float(position.get("last_price", position["entry_price"])),
        )
        value += int(position["shares"]) * float(price)

    return float(value)


def _estimate_position_size(
    cash: float,
    equity: float,
    estimated_entry: float,
    estimated_stop: float,
    portfolio_config: PortfolioConfig,
) -> int:
    estimated_stop_exit = estimated_stop * (
        1 - portfolio_config.slippage_rate
    )

    risk_per_share = (
        estimated_entry * (1 + portfolio_config.commission_rate)
        - estimated_stop_exit
        * (1 - portfolio_config.commission_rate)
    )

    if risk_per_share <= 0:
        return 0

    risk_budget = equity * portfolio_config.risk_per_trade
    shares_by_risk = int(risk_budget / risk_per_share)

    cost_per_share = estimated_entry * (
        1 + portfolio_config.commission_rate
    )
    shares_by_cash = int(cash / cost_per_share)

    if portfolio_config.max_position_fraction is None:
        shares_by_cap = shares_by_cash
    else:
        position_budget = (
            equity
            * portfolio_config.max_position_fraction
        )
        shares_by_cap = int(position_budget / cost_per_share)

    return max(
        0,
        min(
            shares_by_risk,
            shares_by_cash,
            shares_by_cap,
        ),
    )


def create_daily_plan(
    scored_prices: pd.DataFrame,
    state: dict[str, Any],
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    signal_date: str | pd.Timestamp | None = None,
    minimum_coverage_ratio: float = 0.60,
) -> DailyPlan:
    """Create next-open buy/sell instructions from the latest completed bar."""
    required = {
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "LOW_10_PREV",
        "RET_63",
        "RET_126",
        "ADX",
        "RSI",
        "Score",
        "Signal",
        "Reasons",
    }

    missing = required.difference(scored_prices.columns)

    if missing:
        raise KeyError(
            f"Günlük plan için eksik sütunlar: {sorted(missing)}"
        )

    selected_date = (
        pd.Timestamp(signal_date)
        if signal_date is not None
        else choose_signal_date(
            scored_prices,
            minimum_coverage_ratio=minimum_coverage_ratio,
        )
    )

    latest_rows = _latest_rows_on_date(
        scored_prices,
        selected_date,
    )

    if latest_rows.empty:
        raise ValueError(
            f"{selected_date.date()} tarihinde fiyat satırı bulunamadı."
        )

    latest_by_ticker = latest_rows.set_index("Ticker")
    open_positions = set(state["positions"])

    sell_records: list[dict[str, Any]] = []
    hold_records: list[dict[str, Any]] = []

    for ticker, position in state["positions"].items():
        if ticker not in latest_by_ticker.index:
            hold_records.append(
                {
                    "Ticker": ticker,
                    "Action": "HOLD_NO_NEW_BAR",
                    "Reason": "Sinyal tarihinde yeni fiyat barı yok",
                    "Entry_Price": float(position["entry_price"]),
                    "Shares": int(position["shares"]),
                    "Stop_Loss": float(position["stop_loss"]),
                }
            )
            continue

        row = latest_by_ticker.loc[ticker]

        highest_price = max(
            float(position["highest_price"]),
            float(row["High"]),
        )

        close = float(row["Close"])
        low = float(row["Low"])
        atr = float(row["ATR"])
        entry_price = float(position["entry_price"])
        stop_loss = float(position["stop_loss"])
        low_10_prev = float(row["LOW_10_PREV"])

        profit_pct = (close - entry_price) / entry_price
        trailing_level = (
            highest_price
            - strategy_config.trailing_stop_atr * atr
        )

        action = "HOLD"
        reason = ""
        execution = ""

        if low <= stop_loss:
            action = "STOP_TRIGGERED"
            reason = "Gün içi başlangıç stop seviyesi görüldü"
            execution = "Paper kayıtta gerçekleşen fiyatı gir"
        elif (
            profit_pct
            > strategy_config.trailing_activation_return
            and close < trailing_level
        ):
            action = "SELL_NEXT_OPEN"
            reason = "Trailing Stop"
            execution = "Sonraki işlem günü açılışı"
        elif close < low_10_prev:
            action = "SELL_NEXT_OPEN"
            reason = "LOW10 Altı"
            execution = "Sonraki işlem günü açılışı"

        record = {
            "Ticker": ticker,
            "Action": action,
            "Reason": reason,
            "Execution": execution,
            "Entry_Price": entry_price,
            "Close": close,
            "Shares": int(position["shares"]),
            "PnL_%_Before_Cost": profit_pct * 100,
            "Stop_Loss": stop_loss,
            "Highest_Price": highest_price,
            "Trailing_Level": trailing_level,
            "LOW_10_PREV": low_10_prev,
            "ATR": atr,
        }

        if action in {"SELL_NEXT_OPEN", "STOP_TRIGGERED"}:
            sell_records.append(record)
        else:
            hold_records.append(record)

    sell_orders = pd.DataFrame(sell_records)
    hold_positions = pd.DataFrame(hold_records)

    expected_exits = (
        0
        if sell_orders.empty
        else int(
            sell_orders["Action"]
            .isin(["SELL_NEXT_OPEN", "STOP_TRIGGERED"])
            .sum()
        )
    )

    available_slots = max(
        0,
        portfolio_config.max_positions
        - len(open_positions)
        + expected_exits,
    )

    candidate_universe = latest_rows.loc[
        ~latest_rows["Ticker"].isin(open_positions)
    ].copy()

    ranked_candidates = rank_candidates(candidate_universe)

    equity = _portfolio_equity(state, latest_rows)
    estimated_cash = float(state["cash"])

    buy_records: list[dict[str, Any]] = []

    for rank, (_, row) in enumerate(
        ranked_candidates.head(available_slots).iterrows(),
        start=1,
    ):
        estimated_entry = float(row["Close"]) * (
            1 + portfolio_config.slippage_rate
        )

        estimated_stop = (
            estimated_entry
            - strategy_config.initial_stop_atr
            * float(row["ATR"])
        )

        if estimated_stop <= 0:
            continue

        estimated_shares = _estimate_position_size(
            cash=estimated_cash,
            equity=equity,
            estimated_entry=estimated_entry,
            estimated_stop=estimated_stop,
            portfolio_config=portfolio_config,
        )

        if estimated_shares <= 0:
            continue

        estimated_cost = (
            estimated_shares
            * estimated_entry
            * (1 + portfolio_config.commission_rate)
        )

        buy_records.append(
            {
                "Rank": rank,
                "Ticker": row["Ticker"],
                "Action": "BUY_NEXT_OPEN",
                "Signal_Date": selected_date,
                "Reference_Close": float(row["Close"]),
                "Estimated_Entry": estimated_entry,
                "ATR": float(row["ATR"]),
                "Estimated_Stop": estimated_stop,
                "Estimated_Shares": estimated_shares,
                "Estimated_Cost_TL": estimated_cost,
                "Score": int(row["Score"]),
                "RSI": float(row["RSI"]),
                "ADX": float(row["ADX"]),
                "RET_63_%": float(row["RET_63"]) * 100,
                "RET_126_%": float(row["RET_126"]) * 100,
                "Reasons": row["Reasons"],
            }
        )

        estimated_cash -= estimated_cost

    buy_orders = pd.DataFrame(buy_records)

    summary = pd.DataFrame(
        [
            {
                "Signal_Date": selected_date,
                "Universe_Rows": len(latest_rows),
                "Market_Positive": bool(
                    latest_rows["MarketPositive"].iloc[0]
                )
                if "MarketPositive" in latest_rows.columns
                else None,
                "Paper_Cash_TL": float(state["cash"]),
                "Paper_Equity_TL": equity,
                "Open_Positions": len(open_positions),
                "Expected_Exits": expected_exits,
                "Available_Slots": available_slots,
                "Buy_Orders": len(buy_orders),
                "Sell_Orders": len(sell_orders),
                "Hold_Positions": len(hold_positions),
            }
        ]
    )

    return DailyPlan(
        signal_date=selected_date,
        summary=summary,
        buy_orders=buy_orders,
        sell_orders=sell_orders,
        hold_positions=hold_positions,
        ranked_candidates=ranked_candidates,
    )


def save_daily_plan(
    plan: DailyPlan,
    output_directory: str | pd.io.common.BaseBuffer,
) -> dict[str, Any]:
    """Save every daily-plan table into a date-stamped directory."""
    from pathlib import Path

    root = Path(output_directory)
    date_directory = root / plan.signal_date.strftime("%Y-%m-%d")
    date_directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": date_directory / "summary.csv",
        "buys": date_directory / "buy_orders.csv",
        "sells": date_directory / "sell_orders.csv",
        "holds": date_directory / "hold_positions.csv",
        "candidates": date_directory / "ranked_candidates.csv",
    }

    plan.summary.to_csv(paths["summary"], index=False)
    plan.buy_orders.to_csv(paths["buys"], index=False)
    plan.sell_orders.to_csv(paths["sells"], index=False)
    plan.hold_positions.to_csv(paths["holds"], index=False)
    plan.ranked_candidates.to_csv(
        paths["candidates"],
        index=False,
    )

    return paths
