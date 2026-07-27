"""Read the final Robot vs BIST100 backtest artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.utils import dataframe_records


class BacktestService:
    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)

    def _path(self, filename: str) -> Path:
        path = self.results_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Backtest sonucu bulunamadı: {path}. "
                "Önce 09_final_strategy_vs_bist100.ipynb "
                "notebook'unu çalıştır."
            )

        return path

    def summary(self) -> dict[str, Any]:
        periods = pd.read_csv(
            self._path(
                "final_strategy_vs_bist100_periods.csv"
            )
        )
        active = pd.read_csv(
            self._path(
                "final_strategy_vs_bist100_active_metrics.csv"
            )
        )

        return {
            "periods": dataframe_records(periods),
            "active_metrics": dataframe_records(active),
        }

    def equity(self) -> list[dict[str, Any]]:
        strategy = pd.read_parquet(
            self._path("final_strategy_equity.parquet")
        )[["Date", "Equity"]].rename(
            columns={"Equity": "Final_Strategy"}
        )

        gross = pd.read_parquet(
            self._path(
                "bist100_gross_benchmark_equity.parquet"
            )
        )[["Date", "Equity"]].rename(
            columns={"Equity": "BIST100_Gross"}
        )

        net = pd.read_parquet(
            self._path(
                "bist100_net_benchmark_equity.parquet"
            )
        )[["Date", "Equity"]].rename(
            columns={"Equity": "BIST100_Net"}
        )

        merged = (
            strategy.merge(gross, on="Date", how="inner")
            .merge(net, on="Date", how="inner")
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return dataframe_records(merged)

    def yearly(self) -> list[dict[str, Any]]:
        frame = pd.read_csv(
            self._path(
                "final_strategy_vs_bist100_yearly.csv"
            )
        )
        return dataframe_records(frame)

    def monthly_summary(self) -> list[dict[str, Any]]:
        frame = pd.read_csv(
            self._path(
                "final_strategy_vs_bist100_monthly_summary.csv"
            )
        )
        return dataframe_records(frame)
