"""Persistent paper-trading portfolio state and transaction helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PortfolioConfig


STATE_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_paper_state(
    initial_capital: float,
) -> dict[str, Any]:
    """Create an empty paper-trading portfolio."""
    if initial_capital <= 0:
        raise ValueError("Başlangıç sermayesi sıfırdan büyük olmalıdır.")

    return {
        "version": STATE_VERSION,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "initial_capital": float(initial_capital),
        "cash": float(initial_capital),
        "positions": {},
    }


def load_paper_state(
    path: str | Path,
    initial_capital: float | None = None,
) -> dict[str, Any]:
    """Load state or create a new one when the file does not exist."""
    state_path = Path(path)

    if not state_path.exists():
        if initial_capital is None:
            raise FileNotFoundError(
                "Paper-trading state dosyası bulunamadı ve "
                "initial_capital verilmedi."
            )

        state = create_paper_state(initial_capital)
        save_paper_state(state, state_path)
        return state

    with state_path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    required = {
        "version",
        "initial_capital",
        "cash",
        "positions",
    }
    missing = required.difference(state)

    if missing:
        raise ValueError(
            f"State dosyasında eksik alanlar: {sorted(missing)}"
        )

    if not isinstance(state["positions"], dict):
        raise TypeError("'positions' alanı sözlük olmalıdır.")

    return state


def save_paper_state(
    state: dict[str, Any],
    path: str | Path,
) -> Path:
    """Atomically persist the portfolio state as JSON."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    output_state = deepcopy(state)
    output_state["updated_at"] = _utc_now_iso()

    temporary_path = state_path.with_suffix(
        state_path.suffix + ".tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            output_state,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    temporary_path.replace(state_path)
    return state_path


def positions_dataframe(
    state: dict[str, Any],
) -> pd.DataFrame:
    """Convert open positions to a user-friendly DataFrame."""
    records = []

    for ticker, position in state["positions"].items():
        record = {"Ticker": ticker}
        record.update(position)
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "entry_date",
                "entry_price",
                "shares",
                "stop_loss",
                "highest_price",
                "signal_score",
            ]
        )

    return pd.DataFrame(records).sort_values("Ticker")


def calculate_fill_position_size(
    state: dict[str, Any],
    fill_price: float,
    stop_loss: float,
    current_equity: float,
    portfolio_config: PortfolioConfig,
) -> int:
    """Calculate shares from an actual fill price."""
    if fill_price <= 0:
        raise ValueError("fill_price sıfırdan büyük olmalıdır.")

    if stop_loss <= 0 or stop_loss >= fill_price:
        raise ValueError(
            "stop_loss sıfırdan büyük ve fill_price'dan küçük olmalıdır."
        )

    estimated_stop_exit = stop_loss * (
        1 - portfolio_config.slippage_rate
    )

    risk_per_share = (
        fill_price * (1 + portfolio_config.commission_rate)
        - estimated_stop_exit
        * (1 - portfolio_config.commission_rate)
    )

    if risk_per_share <= 0:
        return 0

    risk_budget = current_equity * portfolio_config.risk_per_trade
    shares_by_risk = int(risk_budget / risk_per_share)

    cost_per_share = fill_price * (
        1 + portfolio_config.commission_rate
    )
    shares_by_cash = int(float(state["cash"]) / cost_per_share)

    if portfolio_config.max_position_fraction is None:
        shares_by_cap = shares_by_cash
    else:
        position_budget = (
            current_equity
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


def record_buy(
    state: dict[str, Any],
    ticker: str,
    fill_date: str | pd.Timestamp,
    fill_price: float,
    shares: int,
    stop_loss: float,
    signal_score: int,
    portfolio_config: PortfolioConfig,
) -> dict[str, Any]:
    """Record a paper buy and update cash and open positions."""
    if ticker in state["positions"]:
        raise ValueError(f"{ticker} zaten açık pozisyonlarda.")

    if len(state["positions"]) >= portfolio_config.max_positions:
        raise ValueError("Maksimum pozisyon sayısına ulaşıldı.")

    if shares <= 0:
        raise ValueError("shares sıfırdan büyük olmalıdır.")

    if fill_price <= 0:
        raise ValueError("fill_price sıfırdan büyük olmalıdır.")

    total_cost = (
        shares
        * fill_price
        * (1 + portfolio_config.commission_rate)
    )

    if total_cost > float(state["cash"]) + 1e-8:
        raise ValueError("Yeterli nakit bulunmuyor.")

    state["cash"] = float(state["cash"]) - total_cost

    state["positions"][ticker] = {
        "entry_date": pd.Timestamp(fill_date).isoformat(),
        "entry_price": float(fill_price),
        "shares": int(shares),
        "stop_loss": float(stop_loss),
        "highest_price": float(fill_price),
        "last_price": float(fill_price),
        "signal_score": int(signal_score),
    }

    return state


def record_sell(
    state: dict[str, Any],
    ticker: str,
    fill_date: str | pd.Timestamp,
    fill_price: float,
    reason: str,
    portfolio_config: PortfolioConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record a paper sell and return the closed-trade record."""
    if ticker not in state["positions"]:
        raise KeyError(f"{ticker} açık pozisyonlarda bulunmuyor.")

    if fill_price <= 0:
        raise ValueError("fill_price sıfırdan büyük olmalıdır.")

    position = state["positions"][ticker]
    shares = int(position["shares"])

    gross_value = shares * fill_price
    net_value = gross_value * (
        1 - portfolio_config.commission_rate
    )

    entry_cost = (
        shares
        * float(position["entry_price"])
        * (1 + portfolio_config.commission_rate)
    )

    pnl = net_value - entry_cost
    return_rate = net_value / entry_cost - 1

    state["cash"] = float(state["cash"]) + net_value
    del state["positions"][ticker]

    trade = {
        "Ticker": ticker,
        "Entry_Date": position["entry_date"],
        "Exit_Date": pd.Timestamp(fill_date).isoformat(),
        "Entry_Price": float(position["entry_price"]),
        "Exit_Price": float(fill_price),
        "Shares": shares,
        "Reason": str(reason),
        "PnL_TL": float(pnl),
        "Return": float(return_rate),
        "Return_%": float(return_rate * 100),
        "Signal_Score": int(position["signal_score"]),
    }

    return state, trade


def append_closed_trade(
    trade: dict[str, Any],
    path: str | Path,
) -> Path:
    """Append one closed trade to a CSV journal."""
    trade_path = Path(path)
    trade_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([trade])

    if trade_path.exists():
        new_row.to_csv(
            trade_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        new_row.to_csv(
            trade_path,
            index=False,
        )

    return trade_path


def mark_to_market(
    state: dict[str, Any],
    latest_prices: dict[str, float],
) -> dict[str, float]:
    """Calculate current portfolio equity from supplied latest prices."""
    positions_value = 0.0

    for ticker, position in state["positions"].items():
        price = latest_prices.get(
            ticker,
            float(position.get("last_price", position["entry_price"])),
        )

        positions_value += int(position["shares"]) * float(price)

    cash = float(state["cash"])
    equity = cash + positions_value

    return {
        "Cash": cash,
        "Positions_Value": positions_value,
        "Equity": equity,
        "Open_Positions": len(state["positions"]),
        "Return_%": (
            equity / float(state["initial_capital"]) - 1
        ) * 100,
    }


def append_equity_snapshot(
    snapshot: dict[str, Any],
    path: str | Path,
) -> Path:
    """Append a dated equity snapshot to CSV."""
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(snapshot)
    row.setdefault("Timestamp", _utc_now_iso())

    frame = pd.DataFrame([row])

    if snapshot_path.exists():
        frame.to_csv(
            snapshot_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        frame.to_csv(
            snapshot_path,
            index=False,
        )

    return snapshot_path
