"""Latest BIST100 prices with live-data preference and research fallback."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

import pandas as pd


class MarketService:
    def __init__(
        self,
        live_stock_path: str | Path,
        fallback_stock_path: str | Path | None = None,
    ):
        self.live_path = Path(live_stock_path)
        self.fallback_path = (
            Path(fallback_stock_path)
            if fallback_stock_path is not None
            else None
        )
        self._cached_mtime: float | None = None
        self._cached_path: Path | None = None
        self._cached_rows = pd.DataFrame()

    def _active_path(self) -> Path | None:
        if self.live_path.exists():
            return self.live_path

        if (
            self.fallback_path is not None
            and self.fallback_path.exists()
        ):
            return self.fallback_path

        return None

    def latest_rows(self) -> pd.DataFrame:
        active_path = self._active_path()

        if active_path is None:
            return pd.DataFrame(
                columns=[
                    "Date",
                    "Ticker",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            )

        mtime = active_path.stat().st_mtime

        if (
            self._cached_path == active_path
            and self._cached_mtime == mtime
            and not self._cached_rows.empty
        ):
            return self._cached_rows.copy()

        frame = pd.read_parquet(active_path)
        frame["Date"] = pd.to_datetime(frame["Date"])

        latest = (
            frame.sort_values(["Ticker", "Date"])
            .groupby("Ticker", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

        self._cached_path = active_path
        self._cached_mtime = mtime
        self._cached_rows = latest
        return latest.copy()

    def latest_price_map(self) -> dict[str, float]:
        latest = self.latest_rows()

        if latest.empty:
            return {}

        return dict(
            zip(
                latest["Ticker"].astype(str),
                latest["Close"].astype(float),
                strict=False,
            )
        )

    def status(self) -> dict[str, Any]:
        active_path = self._active_path()
        latest = self.latest_rows()

        return {
            "data_file_exists": active_path is not None,
            "data_file": (
                str(active_path)
                if active_path is not None
                else None
            ),
            "data_source": (
                "live"
                if active_path == self.live_path
                else "research_fallback"
                if active_path is not None
                else "missing"
            ),
            "latest_date": (
                latest["Date"].max().isoformat()
                if not latest.empty
                else None
            ),
            "ticker_count": (
                int(latest["Ticker"].nunique())
                if not latest.empty
                else 0
            ),
        }

    @staticmethod
    def _fetch_single_quote(
        ticker: str,
    ) -> dict[str, Any]:
        """Fetch the newest available Yahoo Finance quote for one ticker."""
        import yfinance as yf

        attempts = [
            {
                "period": "1d",
                "interval": "1m",
                "label": "Yahoo Finance 1m",
            },
            {
                "period": "5d",
                "interval": "5m",
                "label": "Yahoo Finance 5m",
            },
            {
                "period": "5d",
                "interval": "1d",
                "label": "Yahoo Finance daily",
            },
        ]

        errors: list[str] = []

        for attempt in attempts:
            try:
                history = yf.Ticker(ticker).history(
                    period=attempt["period"],
                    interval=attempt["interval"],
                    auto_adjust=False,
                    prepost=False,
                    actions=False,
                )
            except Exception as error:
                errors.append(
                    f"{attempt['interval']}: {error}"
                )
                continue

            if history.empty or "Close" not in history.columns:
                errors.append(
                    f"{attempt['interval']}: veri yok"
                )
                continue

            closes = pd.to_numeric(
                history["Close"],
                errors="coerce",
            ).dropna()

            if closes.empty:
                errors.append(
                    f"{attempt['interval']}: geçerli fiyat yok"
                )
                continue

            quote_time = pd.Timestamp(
                closes.index[-1]
            )

            return {
                "Ticker": ticker,
                "Price": float(closes.iloc[-1]),
                "Quote_Time": quote_time.isoformat(),
                "Source": attempt["label"],
                "Status": "OK",
                "Error": None,
            }

        return {
            "Ticker": ticker,
            "Price": None,
            "Quote_Time": None,
            "Source": "Yahoo Finance",
            "Status": "ERROR",
            "Error": " | ".join(errors)
            or "Fiyat alınamadı.",
        }

    def fetch_latest_quotes(
        self,
        tickers: Iterable[str],
    ) -> list[dict[str, Any]]:
        """Fetch current/latest quotes without rebuilding signals or data."""
        symbols = sorted(
            {
                str(ticker).strip().upper()
                for ticker in tickers
                if str(ticker).strip()
            }
        )

        if not symbols:
            return []

        records: list[dict[str, Any]] = []
        worker_count = min(8, len(symbols))

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = {
                executor.submit(
                    self._fetch_single_quote,
                    ticker,
                ): ticker
                for ticker in symbols
            }

            for future in as_completed(futures):
                ticker = futures[future]

                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "Ticker": ticker,
                        "Price": None,
                        "Quote_Time": None,
                        "Source": "Yahoo Finance",
                        "Status": "ERROR",
                        "Error": str(error),
                    }

                records.append(record)

        return sorted(
            records,
            key=lambda record: record["Ticker"],
        )
