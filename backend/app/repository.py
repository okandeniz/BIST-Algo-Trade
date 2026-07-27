"""Transactional portfolio repository backed by SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from typing import Any

import pandas as pd

from backend.app.database import Database
from backend.app.schemas import BuyRequest, SellRequest
from backend.app.utils import normalize_ticker


BUY_COMMISSION_RATE = 0.002
SELL_COMMISSION_RATE = 0.002


class TradingRepository:
    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def list_portfolios(self) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    name,
                    initial_capital,
                    cash,
                    is_initialized,
                    created_at,
                    updated_at
                FROM portfolios
                ORDER BY name
                """
            ).fetchall()

        records = []

        for row in rows:
            record = dict(row)
            initialized = bool(record["is_initialized"])
            records.append(
                {
                    "name": record["name"],
                    "initial_capital": (
                        float(record["initial_capital"])
                        if initialized
                        else 0.0
                    ),
                    "cash": (
                        float(record["cash"])
                        if initialized
                        else 0.0
                    ),
                    "is_initialized": initialized,
                    "created_at": record["created_at"],
                    "updated_at": record["updated_at"],
                }
            )

        return records

    def _portfolio_row(
        self,
        connection: sqlite3.Connection,
        portfolio_name: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT *
            FROM portfolios
            WHERE name = ?
            """,
            (portfolio_name,),
        ).fetchone()

        if row is None:
            raise KeyError(
                f"Portföy bulunamadı: {portfolio_name}"
            )

        return row

    @staticmethod
    def _require_initialized(
        portfolio: sqlite3.Row,
    ) -> None:
        if not bool(portfolio["is_initialized"]):
            raise ValueError(
                "Önce portföyün başlangıç sermayesini girin."
            )

    def set_initial_capital(
        self,
        portfolio_name: str,
        initial_capital: float,
    ) -> dict[str, Any]:
        """Initialize an empty portfolio with the user's own cash."""
        if initial_capital <= 0:
            raise ValueError(
                "Başlangıç sermayesi sıfırdan büyük olmalıdır."
            )

        now = self._now()

        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                portfolio = self._portfolio_row(
                    connection,
                    portfolio_name,
                )

                if bool(portfolio["is_initialized"]):
                    raise ValueError(
                        "Portföy daha önce başlatılmış. Sermayeyi "
                        "değiştirmek için önce portföyü sıfırlayın."
                    )

                activity = connection.execute(
                    """
                    SELECT
                        (
                            SELECT COUNT(*)
                            FROM positions
                            WHERE portfolio_id = ?
                        ) AS position_count,
                        (
                            SELECT COUNT(*)
                            FROM transactions
                            WHERE portfolio_id = ?
                        ) AS transaction_count
                    """,
                    (portfolio["id"], portfolio["id"]),
                ).fetchone()

                if (
                    int(activity["position_count"]) > 0
                    or int(activity["transaction_count"]) > 0
                ):
                    raise ValueError(
                        "Portföy boş değil. Önce sıfırlama yapın."
                    )

                connection.execute(
                    """
                    UPDATE portfolios
                    SET
                        initial_capital = ?,
                        cash = ?,
                        is_initialized = 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        float(initial_capital),
                        float(initial_capital),
                        now,
                        portfolio["id"],
                    ),
                )
                connection.commit()

                return {
                    "portfolio_name": portfolio_name,
                    "initial_capital": float(initial_capital),
                    "cash": float(initial_capital),
                    "is_initialized": True,
                    "message": (
                        "Portföy başlangıç sermayesiyle oluşturuldu."
                    ),
                }

            except Exception:
                connection.rollback()
                raise

    def reset_portfolio(
        self,
        portfolio_name: str,
    ) -> dict[str, Any]:
        """Delete all activity and require a new capital entry."""
        now = self._now()

        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                portfolio = self._portfolio_row(
                    connection,
                    portfolio_name,
                )

                deleted_positions = connection.execute(
                    """
                    DELETE FROM positions
                    WHERE portfolio_id = ?
                    """,
                    (portfolio["id"],),
                ).rowcount

                deleted_transactions = connection.execute(
                    """
                    DELETE FROM transactions
                    WHERE portfolio_id = ?
                    """,
                    (portfolio["id"],),
                ).rowcount

                # 1.0 is an internal compatibility value for databases
                # created with the old CHECK(initial_capital > 0) schema.
                # The API masks it while is_initialized = 0.
                connection.execute(
                    """
                    UPDATE portfolios
                    SET
                        initial_capital = 1.0,
                        cash = 1.0,
                        is_initialized = 0,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, portfolio["id"]),
                )
                connection.commit()

                return {
                    "portfolio_name": portfolio_name,
                    "is_initialized": False,
                    "deleted_positions": int(
                        deleted_positions or 0
                    ),
                    "deleted_transactions": int(
                        deleted_transactions or 0
                    ),
                    "message": (
                        "Portföy sıfırlandı. Yeni işlem açmadan önce "
                        "başlangıç sermayesini tekrar girin."
                    ),
                }

            except Exception:
                connection.rollback()
                raise

    def buy(self, request: BuyRequest) -> dict[str, Any]:
        ticker = normalize_ticker(request.ticker)
        gross_amount = request.quantity * request.price

        # Commission is recalculated in the backend so the persisted
        # transaction always uses the fixed 0.2% rate.
        commission_rate = BUY_COMMISSION_RATE
        fees = round(
            gross_amount * commission_rate,
            2,
        )
        total_cost = gross_amount + fees
        now = self._now()

        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                portfolio = self._portfolio_row(
                    connection,
                    request.portfolio_name,
                )
                self._require_initialized(portfolio)

                if total_cost > float(portfolio["cash"]) + 1e-9:
                    raise ValueError(
                        "Portföyde alış için yeterli nakit yok."
                    )

                position = connection.execute(
                    """
                    SELECT *
                    FROM positions
                    WHERE portfolio_id = ?
                      AND ticker = ?
                    """,
                    (portfolio["id"], ticker),
                ).fetchone()

                if position is None:
                    new_quantity = request.quantity
                    average_cost = total_cost / request.quantity
                    stop_loss = request.stop_loss
                    highest_price = request.price
                    entry_date = request.trade_date.isoformat()
                    signal_score = request.signal_score
                    realized_pnl = 0.0
                else:
                    old_quantity = int(position["quantity"])
                    new_quantity = old_quantity + request.quantity
                    average_cost = (
                        old_quantity * float(position["avg_cost"])
                        + total_cost
                    ) / new_quantity

                    # Additional buys may tighten the stop, but never loosen it.
                    stop_loss = max(
                        float(position["stop_loss"]),
                        request.stop_loss,
                    )
                    highest_price = max(
                        float(position["highest_price"]),
                        request.price,
                    )
                    entry_date = min(
                        str(position["entry_date"]),
                        request.trade_date.isoformat(),
                    )
                    signal_score = (
                        request.signal_score
                        if request.signal_score is not None
                        else position["signal_score"]
                    )
                    realized_pnl = float(
                        position["realized_pnl"]
                    )

                connection.execute(
                    """
                    INSERT INTO positions (
                        portfolio_id,
                        ticker,
                        quantity,
                        avg_cost,
                        stop_loss,
                        highest_price,
                        entry_date,
                        signal_score,
                        last_price,
                        realized_pnl,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(portfolio_id, ticker)
                    DO UPDATE SET
                        quantity = excluded.quantity,
                        avg_cost = excluded.avg_cost,
                        stop_loss = excluded.stop_loss,
                        highest_price = excluded.highest_price,
                        entry_date = excluded.entry_date,
                        signal_score = excluded.signal_score,
                        last_price = excluded.last_price,
                        realized_pnl = excluded.realized_pnl,
                        updated_at = excluded.updated_at
                    """,
                    (
                        portfolio["id"],
                        ticker,
                        new_quantity,
                        average_cost,
                        stop_loss,
                        highest_price,
                        entry_date,
                        signal_score,
                        request.price,
                        realized_pnl,
                        now,
                    ),
                )

                new_cash = float(portfolio["cash"]) - total_cost
                connection.execute(
                    """
                    UPDATE portfolios
                    SET cash = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_cash, now, portfolio["id"]),
                )

                cursor = connection.execute(
                    """
                    INSERT INTO transactions (
                        portfolio_id,
                        ticker,
                        side,
                        quantity,
                        price,
                        fees,
                        trade_date,
                        gross_amount,
                        net_amount,
                        realized_pnl,
                        stop_loss,
                        note,
                        created_at
                    )
                    VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        portfolio["id"],
                        ticker,
                        request.quantity,
                        request.price,
                        fees,
                        request.trade_date.isoformat(),
                        gross_amount,
                        -total_cost,
                        request.stop_loss,
                        request.note,
                        now,
                    ),
                )

                connection.commit()

                return {
                    "transaction_id": cursor.lastrowid,
                    "portfolio_name": request.portfolio_name,
                    "ticker": ticker,
                    "side": "BUY",
                    "quantity": request.quantity,
                    "price": request.price,
                    "fees": fees,
                    "commission_rate": commission_rate,
                    "gross_amount": gross_amount,
                    "total_cost": total_cost,
                    "remaining_cash": new_cash,
                    "position_quantity": new_quantity,
                    "position_avg_cost": average_cost,
                    "position_stop_loss": stop_loss,
                }

            except Exception:
                connection.rollback()
                raise

    def sell(self, request: SellRequest) -> dict[str, Any]:
        ticker = normalize_ticker(request.ticker)
        gross_amount = request.quantity * request.price

        # Satış komisyonu backend tarafında yeniden hesaplanır.
        # Böylece istemciden farklı bir masraf gönderilse bile
        # nakit ve gerçekleşen K/Z binde 2 ile kaydedilir.
        commission_rate = SELL_COMMISSION_RATE
        fees = round(
            gross_amount * commission_rate,
            2,
        )
        net_proceeds = gross_amount - fees
        now = self._now()

        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                portfolio = self._portfolio_row(
                    connection,
                    request.portfolio_name,
                )
                self._require_initialized(portfolio)

                position = connection.execute(
                    """
                    SELECT *
                    FROM positions
                    WHERE portfolio_id = ?
                      AND ticker = ?
                    """,
                    (portfolio["id"], ticker),
                ).fetchone()

                if position is None:
                    raise ValueError(
                        f"Açık pozisyon bulunamadı: {ticker}"
                    )

                current_quantity = int(position["quantity"])

                if request.quantity > current_quantity:
                    raise ValueError(
                        "Satış adedi açık pozisyon adedini aşamaz."
                    )

                cost_basis = (
                    request.quantity
                    * float(position["avg_cost"])
                )
                realized_pnl = net_proceeds - cost_basis
                remaining_quantity = (
                    current_quantity - request.quantity
                )
                new_cash = float(portfolio["cash"]) + net_proceeds

                if remaining_quantity == 0:
                    connection.execute(
                        """
                        DELETE FROM positions
                        WHERE portfolio_id = ?
                          AND ticker = ?
                        """,
                        (portfolio["id"], ticker),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE positions
                        SET
                            quantity = ?,
                            last_price = ?,
                            realized_pnl = realized_pnl + ?,
                            updated_at = ?
                        WHERE portfolio_id = ?
                          AND ticker = ?
                        """,
                        (
                            remaining_quantity,
                            request.price,
                            realized_pnl,
                            now,
                            portfolio["id"],
                            ticker,
                        ),
                    )

                connection.execute(
                    """
                    UPDATE portfolios
                    SET cash = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_cash, now, portfolio["id"]),
                )

                cursor = connection.execute(
                    """
                    INSERT INTO transactions (
                        portfolio_id,
                        ticker,
                        side,
                        quantity,
                        price,
                        fees,
                        trade_date,
                        gross_amount,
                        net_amount,
                        realized_pnl,
                        stop_loss,
                        note,
                        created_at
                    )
                    VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        portfolio["id"],
                        ticker,
                        request.quantity,
                        request.price,
                        fees,
                        request.trade_date.isoformat(),
                        gross_amount,
                        net_proceeds,
                        realized_pnl,
                        request.note,
                        now,
                    ),
                )

                connection.commit()

                return {
                    "transaction_id": cursor.lastrowid,
                    "portfolio_name": request.portfolio_name,
                    "ticker": ticker,
                    "side": "SELL",
                    "quantity": request.quantity,
                    "price": request.price,
                    "gross_amount": gross_amount,
                    "fees": fees,
                    "commission_rate": commission_rate,
                    "net_proceeds": net_proceeds,
                    "realized_pnl": realized_pnl,
                    "remaining_quantity": remaining_quantity,
                    "remaining_cash": new_cash,
                }

            except Exception:
                connection.rollback()
                raise

    def positions(
        self,
        portfolio_name: str,
        latest_prices: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        latest_prices = latest_prices or {}

        with self.database.connection() as connection:
            portfolio = self._portfolio_row(
                connection,
                portfolio_name,
            )

            if not bool(portfolio["is_initialized"]):
                return []

            rows = connection.execute(
                """
                SELECT *
                FROM positions
                WHERE portfolio_id = ?
                ORDER BY ticker
                """,
                (portfolio["id"],),
            ).fetchall()

        records: list[dict[str, Any]] = []

        for row in rows:
            record = dict(row)
            ticker = str(record["ticker"])
            latest_price = latest_prices.get(ticker)

            if latest_price is None:
                latest_price = (
                    float(record["last_price"])
                    if record["last_price"] is not None
                    else float(record["avg_cost"])
                )

            quantity = int(record["quantity"])
            average_cost = float(record["avg_cost"])
            stop_loss = float(record["stop_loss"])
            market_value = quantity * latest_price
            cost_value = quantity * average_cost
            unrealized_pnl = market_value - cost_value

            records.append(
                {
                    "Ticker": ticker,
                    "Quantity": quantity,
                    "Average_Cost_TL": average_cost,
                    "Latest_Price_TL": latest_price,
                    "Market_Value_TL": market_value,
                    "Unrealized_PnL_TL": unrealized_pnl,
                    "Unrealized_PnL_%": (
                        unrealized_pnl / cost_value * 100
                        if cost_value > 0
                        else 0.0
                    ),
                    "Stop_Loss_TL": stop_loss,
                    "Distance_To_Stop_%": (
                        latest_price / stop_loss - 1
                    ) * 100,
                    "Highest_Price_TL": float(
                        record["highest_price"]
                    ),
                    "Entry_Date": record["entry_date"],
                    "Signal_Score": record["signal_score"],
                }
            )

        return records

    def transactions(
        self,
        portfolio_name: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            portfolio = self._portfolio_row(
                connection,
                portfolio_name,
            )

            if not bool(portfolio["is_initialized"]):
                return []

            rows = connection.execute(
                """
                SELECT
                    id,
                    ticker,
                    side,
                    quantity,
                    price,
                    fees,
                    trade_date,
                    gross_amount,
                    net_amount,
                    realized_pnl,
                    stop_loss,
                    note,
                    created_at
                FROM transactions
                WHERE portfolio_id = ?
                ORDER BY trade_date DESC, id DESC
                LIMIT ?
                """,
                (portfolio["id"], int(limit)),
            ).fetchall()

        return [dict(row) for row in rows]

    def summary(
        self,
        portfolio_name: str,
        latest_prices: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        latest_prices = latest_prices or {}

        with self.database.connection() as connection:
            portfolio = self._portfolio_row(
                connection,
                portfolio_name,
            )
            initialized = bool(
                portfolio["is_initialized"]
            )

            if not initialized:
                return {
                    "Portfolio": portfolio_name,
                    "Is_Initialized": False,
                    "Initial_Capital_TL": 0.0,
                    "Cash_TL": 0.0,
                    "Positions_Value_TL": 0.0,
                    "Equity_TL": 0.0,
                    "Total_Return_TL": 0.0,
                    "Total_Return_%": 0.0,
                    "Realized_PnL_TL": 0.0,
                    "Unrealized_PnL_TL": 0.0,
                    "Open_Positions": 0,
                    "Transaction_Count": 0,
                    "Invested_%": 0.0,
                }

            activity_row = connection.execute(
                """
                SELECT
                    COALESCE(
                        SUM(
                            CASE
                                WHEN side = 'SELL'
                                THEN realized_pnl
                                ELSE 0
                            END
                        ),
                        0
                    ) AS realized_total,
                    COUNT(*) AS transaction_count
                FROM transactions
                WHERE portfolio_id = ?
                """,
                (portfolio["id"],),
            ).fetchone()

        positions = self.positions(
            portfolio_name,
            latest_prices=latest_prices,
        )

        positions_value = sum(
            float(row["Market_Value_TL"])
            for row in positions
        )
        unrealized_pnl = sum(
            float(row["Unrealized_PnL_TL"])
            for row in positions
        )
        cash = float(portfolio["cash"])
        initial_capital = float(
            portfolio["initial_capital"]
        )
        equity = cash + positions_value

        return {
            "Portfolio": portfolio_name,
            "Is_Initialized": True,
            "Initial_Capital_TL": initial_capital,
            "Cash_TL": cash,
            "Positions_Value_TL": positions_value,
            "Equity_TL": equity,
            "Total_Return_TL": equity - initial_capital,
            "Total_Return_%": (
                equity / initial_capital - 1
            ) * 100,
            "Realized_PnL_TL": float(
                activity_row["realized_total"]
            ),
            "Unrealized_PnL_TL": unrealized_pnl,
            "Open_Positions": len(positions),
            "Transaction_Count": int(
                activity_row["transaction_count"]
            ),
            "Invested_%": (
                positions_value / equity * 100
                if equity > 0
                else 0.0
            ),
        }

    def update_last_prices(
        self,
        portfolio_name: str,
        prices: dict[str, float],
    ) -> int:
        """Update only the stored last price of current open positions."""
        if not prices:
            return 0

        now = self._now()
        updated_count = 0

        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                portfolio = self._portfolio_row(
                    connection,
                    portfolio_name,
                )
                self._require_initialized(portfolio)

                for ticker, price in prices.items():
                    numeric_price = float(price)

                    if numeric_price <= 0:
                        continue

                    cursor = connection.execute(
                        """
                        UPDATE positions
                        SET
                            last_price = ?,
                            updated_at = ?
                        WHERE portfolio_id = ?
                          AND ticker = ?
                        """,
                        (
                            numeric_price,
                            now,
                            portfolio["id"],
                            str(ticker),
                        ),
                    )
                    updated_count += int(
                        cursor.rowcount or 0
                    )

                connection.commit()
                return updated_count

            except Exception:
                connection.rollback()
                raise

    def update_marks(
        self,
        portfolio_name: str,
        marks: pd.DataFrame,
    ) -> None:
        required = {"Ticker", "High", "Close"}
        missing = required.difference(marks.columns)

        if missing:
            raise KeyError(
                f"Pozisyon mark güncellemesi için eksik: "
                f"{sorted(missing)}"
            )

        now = self._now()

        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                portfolio = self._portfolio_row(
                    connection,
                    portfolio_name,
                )

                if not bool(portfolio["is_initialized"]):
                    connection.rollback()
                    return

                for row in marks.itertuples(index=False):
                    connection.execute(
                        """
                        UPDATE positions
                        SET
                            highest_price = MAX(
                                highest_price,
                                ?
                            ),
                            last_price = ?,
                            updated_at = ?
                        WHERE portfolio_id = ?
                          AND ticker = ?
                        """,
                        (
                            float(row.High),
                            float(row.Close),
                            now,
                            portfolio["id"],
                            str(row.Ticker),
                        ),
                    )

                connection.commit()

            except Exception:
                connection.rollback()
                raise

    def as_paper_state(
        self,
        portfolio_name: str,
    ) -> dict[str, Any]:
        with self.database.connection() as connection:
            portfolio = self._portfolio_row(
                connection,
                portfolio_name,
            )

            if not bool(portfolio["is_initialized"]):
                return {
                    "initial_capital": 0.0,
                    "cash": 0.0,
                    "positions": {},
                }

            rows = connection.execute(
                """
                SELECT *
                FROM positions
                WHERE portfolio_id = ?
                ORDER BY ticker
                """,
                (portfolio["id"],),
            ).fetchall()

        positions = {}

        for row in rows:
            positions[str(row["ticker"])] = {
                "entry_date": row["entry_date"],
                "entry_price": float(row["avg_cost"]),
                "shares": int(row["quantity"]),
                "stop_loss": float(row["stop_loss"]),
                "highest_price": float(
                    row["highest_price"]
                ),
                "last_price": (
                    float(row["last_price"])
                    if row["last_price"] is not None
                    else float(row["avg_cost"])
                ),
                "signal_score": int(
                    row["signal_score"] or 0
                ),
            }

        return {
            "initial_capital": float(
                portfolio["initial_capital"]
            ),
            "cash": float(portfolio["cash"]),
            "positions": positions,
        }
