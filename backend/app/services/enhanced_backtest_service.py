"""Generate the RS126 Enhanced audit comparison for the dashboard.

The full-history Baseline vs BIST100 section remains unchanged because the
RS126 rule was selected after Development and Validation research. The fair
comparison is the common 2025+ Audit / walk-forward period.
"""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

import pandas as pd

from backend.app.utils import dataframe_records
from src.product_config import ENHANCED_PORTFOLIO
from src.release_manifest import ReleaseManifest


class EnhancedBacktestService:
    """Build and cache the accepted RS126 Enhanced audit backtest."""

    def __init__(
        self,
        release_manifest: ReleaseManifest,
    ):
        self.release_manifest = release_manifest

    def _walkforward_metadata(self) -> dict[str, Any]:
        path = self.release_manifest.resolve(
            "walk_forward.metadata",
            must_exist=False,
        )

        if not path.exists():
            return {
                "start": "2025-01-01",
                "end": None,
                "warning": (
                    "Walk-forward metadata bulunamadığı için "
                    "RS126 Audit dönemi 2025-01-01'den başlatıldı."
                ),
            }

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    @lru_cache(maxsize=1)
    def _artifacts(
        self,
    ) -> tuple[
        dict[str, Any],
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Return metadata, metrics and equity; cache until API restart."""
        from src.backtest import (
            run_portfolio_backtest,
        )
        from src.enhanced_baseline import (
            RULE_DESCRIPTION,
            RULE_NAME,
            build_rs126_enhanced_prices,
        )
        from src.features import add_indicators
        from src.metrics import (
            portfolio_metrics,
        )
        from src.presets import (
            FINAL_PORTFOLIO_CONFIG,
            FINAL_STRATEGY_CONFIG,
        )
        from src.signals import (
            add_robot_scores,
            build_market_regime,
        )

        stock_path = self.release_manifest.resolve(
            "research.stock_prices"
        )
        market_path = self.release_manifest.resolve(
            "research.market_prices"
        )

        stock_prices = pd.read_parquet(
            stock_path
        )
        market_prices = pd.read_parquet(
            market_path
        )

        stock_features = add_indicators(
            stock_prices
        )
        market_features = add_indicators(
            market_prices
        )
        market_regime = build_market_regime(
            market_features
        )

        baseline_prices = add_robot_scores(
            stock_features=stock_features,
            market_regime=market_regime,
            config=FINAL_STRATEGY_CONFIG,
            include_reasons=False,
        )

        enhanced_prices = (
            build_rs126_enhanced_prices(
                baseline_prices=baseline_prices,
                market_features=market_features,
                strategy_config=(
                    FINAL_STRATEGY_CONFIG
                ),
            )
        )

        wf_metadata = (
            self._walkforward_metadata()
        )

        requested_start = pd.Timestamp(
            wf_metadata.get(
                "start",
                "2025-01-01",
            )
        )

        data_end = min(
            pd.to_datetime(
                enhanced_prices["Date"]
            ).max(),
            pd.to_datetime(
                market_features["Date"]
            ).max(),
        )

        configured_end = wf_metadata.get(
            "end"
        )
        requested_end = (
            pd.Timestamp(configured_end)
            if configured_end
            else data_end
        )
        effective_end = min(
            requested_end,
            data_end,
        )

        period_prices = enhanced_prices.loc[
            pd.to_datetime(
                enhanced_prices["Date"]
            ).between(
                requested_start,
                effective_end,
                inclusive="both",
            )
        ].copy()

        if period_prices.empty:
            raise ValueError(
                "RS126 Enhanced Audit döneminde "
                "backtest verisi bulunamadı."
            )

        equity, trades = (
            run_portfolio_backtest(
                scored_prices=period_prices,
                strategy_config=(
                    FINAL_STRATEGY_CONFIG
                ),
                portfolio_config=(
                    FINAL_PORTFOLIO_CONFIG
                ),
            )
        )

        metrics = portfolio_metrics(
            equity,
            trades,
        )
        metrics["Portfolio"] = ENHANCED_PORTFOLIO
        metrics_frame = pd.DataFrame(
            [metrics]
        )

        equity_frame = (
            equity[
                [
                    "Date",
                    "Equity",
                ]
            ]
            .rename(
                columns={
                    "Equity": ENHANCED_PORTFOLIO
                }
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        metadata = {
            "portfolio": ENHANCED_PORTFOLIO,
            "rule_name": RULE_NAME,
            "rule": RULE_DESCRIPTION,
            "start": (
                requested_start.strftime(
                    "%Y-%m-%d"
                )
            ),
            "end": (
                effective_end.strftime(
                    "%Y-%m-%d"
                )
            ),
            "comparison_scope": (
                "Common Audit / walk-forward period"
            ),
            "full_history_included": False,
            "warning": (
                "RS126 kuralı araştırma dönemlerinden sonra "
                "seçildiği için tam geçmiş grafiğine eklenmez. "
                "Adil kıyaslama 2025+ ortak Audit döneminde yapılır."
            ),
        }

        return (
            metadata,
            metrics_frame,
            equity_frame,
        )

    def summary(self) -> dict[str, Any]:
        metadata, metrics, _ = (
            self._artifacts()
        )

        return {
            "metadata": metadata,
            "metrics": dataframe_records(
                metrics
            ),
        }

    def equity(self) -> list[dict[str, Any]]:
        _, _, equity = self._artifacts()
        return dataframe_records(equity)

    def yearly(self) -> list[dict[str, Any]]:
        """Calculate annual returns from the cached enhanced equity."""
        from src.metrics import yearly_performance

        _, _, equity = self._artifacts()

        working = equity.rename(
            columns={
                ENHANCED_PORTFOLIO: "Equity"
            }
        ).copy()
        working["Date"] = pd.to_datetime(
            working["Date"]
        )
        working["Cash"] = 0.0
        working["Open_Positions"] = 0

        yearly = yearly_performance(
            working
        )[
            [
                "Year",
                "Return_%",
            ]
        ].rename(
            columns={
                "Return_%": (
                    ENHANCED_PORTFOLIO
                )
            }
        )

        return dataframe_records(yearly)

    def refresh(self) -> dict[str, Any]:
        """Clear the in-memory cache and rebuild on the next request."""
        self._artifacts.cache_clear()

        return {
            "message": (
                "RS126 Enhanced backtest önbelleği temizlendi."
            )
        }
