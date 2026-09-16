"""Read the final Robot vs BIST100 backtest artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.utils import dataframe_records
from src.release_manifest import ReleaseManifest


class BacktestService:
    def __init__(self, release_manifest: ReleaseManifest):
        self.release_manifest = release_manifest

    def summary(self) -> dict[str, Any]:
        periods = pd.read_csv(
            self.release_manifest.resolve("baseline.periods")
        )
        active = pd.read_csv(
            self.release_manifest.resolve("baseline.active_metrics")
        )

        return {
            "periods": dataframe_records(periods),
            "active_metrics": dataframe_records(active),
        }

    def equity(self) -> list[dict[str, Any]]:
        strategy = pd.read_parquet(
            self.release_manifest.resolve("baseline.strategy_equity")
        )[["Date", "Equity"]].rename(
            columns={"Equity": "Final_Strategy"}
        )

        gross = pd.read_parquet(
            self.release_manifest.resolve(
                "baseline.benchmark_gross_equity"
            )
        )[["Date", "Equity"]].rename(
            columns={"Equity": "BIST100_Gross"}
        )

        net = pd.read_parquet(
            self.release_manifest.resolve(
                "baseline.benchmark_net_equity"
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
            self.release_manifest.resolve("baseline.yearly")
        )
        return dataframe_records(frame)

    def monthly_summary(self) -> list[dict[str, Any]]:
        frame = pd.read_csv(
            self.release_manifest.resolve("baseline.monthly_summary")
        )
        return dataframe_records(frame)
