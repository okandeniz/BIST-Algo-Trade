"""Event-driven Robot portfolio backtest.

The strategy rules reproduce Robot.ipynb. Execution timing is corrected:
close-based signals are executed at the next available bar's open.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from src.config import PortfolioConfig, StrategyConfig
from src.signals import rank_candidates


def _next_bar_date(df: pd.DataFrame, current_position: int) -> pd.Timestamp | None:
    next_position = current_position + 1
    if next_position >= len(df):
        return None
    return pd.Timestamp(df.iloc[next_position]["Date"])


def _return_and_outlier(
    entry_price: float,
    exit_price: float,
    portfolio_config: PortfolioConfig,
) -> tuple[float, bool]:
    ret = (
        exit_price * (1 - portfolio_config.commission_rate)
        / (entry_price * (1 + portfolio_config.commission_rate))
        - 1
    )
    is_outlier = not (
        portfolio_config.minimum_valid_return
        <= ret
        <= portfolio_config.maximum_valid_return
    )
    return ret, is_outlier


def run_portfolio_backtest(
    scored_prices: pd.DataFrame,
    strategy_config: StrategyConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the corrected multi-asset Robot backtest."""
    strategy_config = strategy_config or StrategyConfig()
    portfolio_config = portfolio_config or PortfolioConfig()

    required = {
        "Date", "Ticker", "Open", "High", "Low", "Close", "ATR",
        "LOW_10_PREV", "RET_63", "RET_126", "ADX", "RSI",
        "Score", "Signal",
    }
    missing = required.difference(scored_prices.columns)
    if missing:
        raise KeyError(f"Backtest için eksik sütunlar: {sorted(missing)}")

    data_dict: dict[str, pd.DataFrame] = {}
    date_position: dict[str, dict[pd.Timestamp, int]] = {}

    for ticker, ticker_df in scored_prices.groupby("Ticker", sort=False):
        work = (
            ticker_df.sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )
        data_dict[ticker] = work
        date_position[ticker] = {
            pd.Timestamp(date): position
            for position, date in enumerate(work["Date"])
        }

    all_dates = sorted(
        set().union(*[set(frame["Date"]) for frame in data_dict.values()])
    )

    cash = portfolio_config.initial_capital
    positions: dict[str, dict[str, Any]] = {}
    pending_entries: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []

    last_close: dict[str, float] = {}

    def current_equity(date: pd.Timestamp, use_open: bool = False) -> float:
        value = cash
        for symbol, position in positions.items():
            frame = data_dict[symbol]
            pos = date_position[symbol].get(date)
            if pos is not None:
                column = "Open" if use_open else "Close"
                mark = float(frame.iloc[pos][column])
            else:
                mark = float(position["last_price"])
            value += int(position["shares"]) * mark
        return float(value)

    def close_position(
        symbol: str,
        date: pd.Timestamp,
        exit_price: float,
        reason: str,
    ) -> None:
        nonlocal cash
        position = positions[symbol]
        net_value = (
            position["shares"]
            * exit_price
            * (1 - portfolio_config.commission_rate)
        )
        cash += net_value

        ret, is_outlier = _return_and_outlier(
            position["entry_price"],
            exit_price,
            portfolio_config,
        )

        trades.append({
            "Ticker": symbol,
            "Entry_Date": position["entry_date"],
            "Exit_Date": date,
            "Entry": position["entry_price"],
            "Exit": exit_price,
            "Shares": position["shares"],
            "Return": ret,
            "Return_%": ret * 100,
            "Reason": reason,
            "Is_Outlier": is_outlier,
            "Signal_Score": position["signal_score"],
        })
        del positions[symbol]

    for current_date in all_dates:
        current_date = pd.Timestamp(current_date)

        # 1) Prior close-based exits execute at today's open.
        for symbol in list(positions):
            position = positions[symbol]
            if not position.get("pending_exit_reason"):
                continue

            pos = date_position[symbol].get(current_date)
            if pos is None or current_date <= position["pending_exit_trigger_date"]:
                continue

            row = data_dict[symbol].iloc[pos]
            exit_price = float(row["Open"]) * (
                1 - portfolio_config.slippage_rate
            )
            reason = str(position["pending_exit_reason"])
            close_position(symbol, current_date, exit_price, reason)

        # 2) Pending buy signals execute at today's open.
        todays_entries = [
            order for order in pending_entries
            if pd.Timestamp(order["Entry_Date"]) == current_date
        ]
        pending_entries = [
            order for order in pending_entries
            if pd.Timestamp(order["Entry_Date"]) != current_date
        ]

        for order in todays_entries:
            symbol = str(order["Ticker"])
            if symbol in positions:
                continue
            if len(positions) >= portfolio_config.max_positions:
                continue

            pos = date_position[symbol].get(current_date)
            if pos is None:
                continue

            row = data_dict[symbol].iloc[pos]
            raw_open = float(row["Open"])
            entry_price = raw_open * (1 + portfolio_config.slippage_rate)
            stop_distance = (
                strategy_config.initial_stop_atr * float(order["Signal_ATR"])
            )

            if entry_price <= 0 or stop_distance <= 0:
                continue

            stop_loss = entry_price - stop_distance
            if stop_loss <= 0:
                continue

            equity_at_open = current_equity(current_date, use_open=True)
            risk_amount = equity_at_open * portfolio_config.risk_per_trade
            estimated_stop_exit = stop_loss * (
                1 - portfolio_config.slippage_rate
            )

            risk_per_share = (
                entry_price * (1 + portfolio_config.commission_rate)
                - estimated_stop_exit
                * (1 - portfolio_config.commission_rate)
            )
            if risk_per_share <= 0:
                continue

            shares_by_risk = int(risk_amount / risk_per_share)

            cost_per_share = (
                entry_price
                * (1 + portfolio_config.commission_rate)
            )
            shares_by_cash = int(cash / cost_per_share)

            if portfolio_config.max_position_fraction is None:
                shares_by_position_cap = shares_by_cash
            else:
                if not (
                    0 < portfolio_config.max_position_fraction <= 1
                ):
                    raise ValueError(
                        "max_position_fraction 0 ile 1 arasında "
                        "olmalı veya None olmalıdır."
                    )

                position_budget = (
                    equity_at_open
                    * portfolio_config.max_position_fraction
                )
                shares_by_position_cap = int(
                    position_budget / cost_per_share
                )

            shares = min(
                shares_by_risk,
                shares_by_cash,
                shares_by_position_cap,
            )

            if shares <= 0:
                continue

            cost = (
                shares
                * entry_price
                * (1 + portfolio_config.commission_rate)
            )
            cash -= cost

            positions[symbol] = {
                "entry_date": current_date,
                "entry_price": entry_price,
                "shares": shares,
                "stop_loss": stop_loss,
                "highest_price": entry_price,
                "last_price": entry_price,
                "signal_score": int(order["Score"]),
                "pending_exit_reason": None,
                "pending_exit_trigger_date": None,
            }

        # 3) Intraday hard-stop logic and close-based exit triggers.
        for symbol in list(positions):
            position = positions[symbol]
            pos = date_position[symbol].get(current_date)
            if pos is None:
                continue

            row = data_dict[symbol].iloc[pos]
            open_price = float(row["Open"])
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            atr = float(row["ATR"])
            low_10_prev = float(row["LOW_10_PREV"])

            position["last_price"] = close
            position["highest_price"] = max(
                float(position["highest_price"]),
                high,
            )
            last_close[symbol] = close

            # Newly opened positions are also exposed to the day's range.
            if open_price <= position["stop_loss"]:
                exit_price = open_price * (
                    1 - portfolio_config.slippage_rate
                )
                close_position(
                    symbol, current_date, exit_price, "Stop Loss Gap"
                )
                continue

            if low <= position["stop_loss"]:
                exit_price = position["stop_loss"] * (
                    1 - portfolio_config.slippage_rate
                )
                close_position(symbol, current_date, exit_price, "Stop Loss")
                continue

            trailing_stop = (
                position["highest_price"]
                - strategy_config.trailing_stop_atr * atr
            )
            profit_pct = (
                close - position["entry_price"]
            ) / position["entry_price"]

            if (
                profit_pct > strategy_config.trailing_activation_return
                and close < trailing_stop
            ):
                position["pending_exit_reason"] = "Trailing Stop"
                position["pending_exit_trigger_date"] = current_date
            elif close < low_10_prev:
                position["pending_exit_reason"] = "LOW10 Altı"
                position["pending_exit_trigger_date"] = current_date

        # 4) Generate close signals and schedule next-bar entries.
        pending_symbols = {str(order["Ticker"]) for order in pending_entries}
        expected_exits = sum(
            bool(position.get("pending_exit_reason"))
            for position in positions.values()
        )
        available_slots = (
            portfolio_config.max_positions
            - (len(positions) - expected_exits)
            - len(pending_entries)
        )

        if available_slots > 0:
            daily_rows = []
            for symbol, frame in data_dict.items():
                if symbol in positions or symbol in pending_symbols:
                    continue

                pos = date_position[symbol].get(current_date)
                if pos is None:
                    continue

                next_date = _next_bar_date(frame, pos)
                if next_date is None:
                    continue

                daily_rows.append(frame.iloc[pos])

            if daily_rows:
                daily_df = pd.DataFrame(daily_rows)
                candidates = rank_candidates(daily_df)

                for _, row in candidates.head(available_slots).iterrows():
                    symbol = str(row["Ticker"])
                    pos = date_position[symbol][current_date]
                    next_date = _next_bar_date(data_dict[symbol], pos)
                    if next_date is None:
                        continue

                    pending_entries.append({
                        "Ticker": symbol,
                        "Signal_Date": current_date,
                        "Entry_Date": next_date,
                        "Score": int(row["Score"]),
                        "RSI": float(row["RSI"]),
                        "ADX": float(row["ADX"]),
                        "RET_63": float(row["RET_63"]),
                        "RET_126": float(row["RET_126"]),
                        "Signal_ATR": float(row["ATR"]),
                    })

        equity_curve.append({
            "Date": current_date,
            "Equity": current_equity(current_date, use_open=False),
            "Cash": cash,
            "Open_Positions": len(positions),
            "Pending_Entries": len(pending_entries),
        })

    # 5) Liquidate remaining positions at their last close, including costs.
    final_date = pd.Timestamp(all_dates[-1])
    for symbol in list(positions):
        position = positions[symbol]
        exit_price = float(position["last_price"]) * (
            1 - portfolio_config.slippage_rate
        )
        close_position(symbol, final_date, exit_price, "Final Close")

    if equity_curve:
        equity_curve[-1]["Equity"] = cash
        equity_curve[-1]["Cash"] = cash
        equity_curve[-1]["Open_Positions"] = 0

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)

    equity_df.attrs["strategy_config"] = asdict(strategy_config)
    equity_df.attrs["portfolio_config"] = asdict(portfolio_config)

    return equity_df, trades_df
