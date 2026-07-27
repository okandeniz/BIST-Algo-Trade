"""SQLite database initialization and connection handling."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    initial_capital REAL NOT NULL CHECK(initial_capital >= 0),
    cash REAL NOT NULL CHECK(cash >= 0),
    is_initialized INTEGER NOT NULL DEFAULT 0
        CHECK(is_initialized IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    portfolio_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    avg_cost REAL NOT NULL CHECK(avg_cost > 0),
    stop_loss REAL NOT NULL CHECK(stop_loss > 0),
    highest_price REAL NOT NULL CHECK(highest_price > 0),
    entry_date TEXT NOT NULL,
    signal_score INTEGER,
    last_price REAL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (portfolio_id, ticker),
    FOREIGN KEY (portfolio_id)
        REFERENCES portfolios(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    price REAL NOT NULL CHECK(price > 0),
    fees REAL NOT NULL DEFAULT 0 CHECK(fees >= 0),
    trade_date TEXT NOT NULL,
    gross_amount REAL NOT NULL,
    net_amount REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    stop_loss REAL,
    note TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (portfolio_id)
        REFERENCES portfolios(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_date
ON transactions(portfolio_id, trade_date DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_transactions_ticker
ON transactions(ticker);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.connection() as connection:
            connection.executescript(SCHEMA_SQL)

            # Migration for databases created before user-defined capital.
            portfolio_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(portfolios)"
                ).fetchall()
            }

            if "is_initialized" not in portfolio_columns:
                connection.execute(
                    """
                    ALTER TABLE portfolios
                    ADD COLUMN is_initialized INTEGER
                    NOT NULL DEFAULT 0
                    """
                )

            now = self._now()

            # Internal value 1.0 stays compatible with old databases whose
            # original CHECK constraint required initial_capital > 0.
            # It is never shown while is_initialized = 0.
            for portfolio_name in (
                "Baseline_Robot",
                "ML_Challenger",
            ):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO portfolios (
                        name,
                        initial_capital,
                        cash,
                        is_initialized,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (
                        portfolio_name,
                        1.0,
                        1.0,
                        now,
                        now,
                    ),
                )

            # Preserve an existing portfolio if it already contains activity.
            connection.execute(
                """
                UPDATE portfolios
                SET is_initialized = 1
                WHERE EXISTS (
                    SELECT 1
                    FROM positions
                    WHERE positions.portfolio_id = portfolios.id
                )
                OR EXISTS (
                    SELECT 1
                    FROM transactions
                    WHERE transactions.portfolio_id = portfolios.id
                )
                """
            )

            connection.commit()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")

        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
