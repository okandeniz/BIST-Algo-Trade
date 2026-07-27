"""Daily Baseline and ML Challenger signal generation and persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import AppSettings
from backend.app.repository import TradingRepository
from backend.app.utils import dataframe_records


class SignalService:
    def __init__(
        self,
        settings: AppSettings,
        repository: TradingRepository,
    ):
        self.settings = settings
        self.repository = repository

    def _latest_plan_directory(self) -> Path:
        """Return the most recently generated valid dual-plan directory.

        The model-ready signal date can be one day behind the latest stock
        date. Therefore the newest folder name is not necessarily the newest
        valid API plan. Old notebook runs may also leave a later-dated folder
        without metadata. Prefer directories that contain metadata.json and
        choose them by generated_at_utc.
        """
        root = self.settings.daily_plans_dir

        if not root.exists():
            raise FileNotFoundError(
                "Henüz günlük plan bulunmuyor. "
                "Sinyalleri yenile düğmesini kullan."
            )

        directories = [
            path
            for path in root.iterdir()
            if path.is_dir()
        ]

        if not directories:
            raise FileNotFoundError(
                "Günlük plan klasörü boş."
            )

        valid_candidates: list[
            tuple[pd.Timestamp, Path]
        ] = []

        for directory in directories:
            metadata_path = directory / "metadata.json"
            summary_path = directory / "dual_summary.csv"

            if (
                not metadata_path.exists()
                or not summary_path.exists()
            ):
                continue

            try:
                metadata = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )
                generated_at = pd.Timestamp(
                    metadata.get(
                        "generated_at_utc",
                        metadata.get(
                            "signal_date",
                            directory.name,
                        ),
                    )
                )
            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                continue

            valid_candidates.append(
                (generated_at, directory)
            )

        if valid_candidates:
            return max(
                valid_candidates,
                key=lambda item: item[0],
            )[1]

        # Backward-compatible fallback for plans created before metadata was
        # introduced. This should only be used until the first API refresh.
        return max(directories, key=lambda path: path.name)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, Any]]:
        """Read a plan CSV safely.

        A plan with no buy or sell orders can be persisted as a zero-byte
        CSV file. Pandas raises EmptyDataError for that file. The API should
        return an empty list instead of failing.
        """
        if not path.exists() or path.stat().st_size == 0:
            return []

        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return []

        return dataframe_records(frame)

    def latest(self) -> dict[str, Any]:
        directory = self._latest_plan_directory()
        metadata_path = directory / "metadata.json"

        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

        return {
            "signal_date": (
                str(metadata.get("signal_date"))
                if metadata.get("signal_date")
                else directory.name
            ),
            "metadata": metadata,
            "summary": self._read_csv(
                directory / "dual_summary.csv"
            ),
            "buy_comparison": self._read_csv(
                directory / "dual_buy_comparison.csv"
            ),
            "sell_comparison": self._read_csv(
                directory / "dual_sell_comparison.csv"
            ),
            "baseline_buys": self._read_csv(
                directory / "baseline_buy_orders.csv"
            ),
            "baseline_sells": self._read_csv(
                directory / "baseline_sell_orders.csv"
            ),
            "challenger_buys": self._read_csv(
                directory / "challenger_buy_orders.csv"
            ),
            "challenger_sells": self._read_csv(
                directory / "challenger_sell_orders.csv"
            ),
            "model_diagnostic": self._read_csv(
                directory / "model_diagnostic.csv"
            ),
            "source_directory": str(directory),
        }

    def refresh(self) -> dict[str, Any]:
        """Download current data and produce synchronized dual plans."""
        # Imports remain local so API startup can still work when a model
        # artifact or research dependency is temporarily missing.
        from src.config import DataConfig
        from src.data_loader import (
            download_robot_bundle,
            load_bist_tickers,
        )
        from src.data_quality import run_quality_pipeline
        from src.features import add_indicators
        from src.signals import (
            add_robot_scores,
            build_market_regime,
        )
        from src.presets import (
            FINAL_PORTFOLIO_CONFIG,
            FINAL_STRATEGY_CONFIG,
        )
        from src.ml_dataset import add_meta_features
        from src.dual_paper import (
            assert_model_scores_available,
            build_challenger_prices,
            choose_model_ready_signal_date,
            create_dual_daily_plan,
            load_challenger_deployment,
            model_score_diagnostic,
            save_dual_plans,
        )

        if not self.settings.ticker_file.exists():
            raise FileNotFoundError(
                f"Ticker dosyası bulunamadı: "
                f"{self.settings.ticker_file}"
            )

        start = (
            pd.Timestamp.today().normalize()
            - pd.Timedelta(
                days=self.settings.signal_lookback_days
            )
        ).strftime("%Y-%m-%d")

        data_config = DataConfig(
            start=start,
            end=None,
            auto_adjust=True,
            yfinance_repair=False,
        )

        tickers = load_bist_tickers(
            self.settings.ticker_file
        )

        raw_stocks, raw_market, errors = (
            download_robot_bundle(
                tickers=tickers,
                config=data_config,
            )
        )

        stock_quality = run_quality_pipeline(
            raw_stocks,
            config=data_config,
            apply_split_repairs=True,
        )
        market_quality = run_quality_pipeline(
            raw_market,
            config=data_config,
            apply_split_repairs=True,
        )

        # Daily API data is operational/live data. It must never overwrite
        # the full-history research parquet files under data/processed.
        live_dir = self.settings.live_stock_path.parent
        live_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.settings.app_results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_stocks.to_parquet(
            live_dir / "bist100_live_raw.parquet",
            index=False,
        )
        raw_market.to_parquet(
            live_dir / "xu100_live_raw.parquet",
            index=False,
        )
        stock_quality.clean.to_parquet(
            self.settings.live_stock_path,
            index=False,
        )
        market_quality.clean.to_parquet(
            self.settings.live_market_path,
            index=False,
        )
        errors.to_csv(
            self.settings.app_results_dir
            / "daily_download_errors.csv",
            index=False,
        )
        stock_quality.summary.to_csv(
            self.settings.app_results_dir
            / "daily_data_quality_summary.csv",
            index=False,
        )

        stock_features = add_indicators(
            stock_quality.clean
        )
        market_features = add_indicators(
            market_quality.clean
        )
        market_regime = build_market_regime(
            market_features
        )

        baseline_prices = add_robot_scores(
            stock_features=stock_features,
            market_regime=market_regime,
            config=FINAL_STRATEGY_CONFIG,
            include_reasons=True,
        )

        featured_prices = add_meta_features(
            scored_prices=baseline_prices,
            market_features=market_features,
        )

        deployment, model = load_challenger_deployment(
            self.settings.project_root
        )

        signal_date = choose_model_ready_signal_date(
            featured_prices=featured_prices,
            minimum_coverage_ratio=0.60,
        )
        latest_stock_date = pd.Timestamp(
            baseline_prices["Date"].max()
        )

        challenger_prices = build_challenger_prices(
            featured_prices=featured_prices,
            fitted_model=model,
            deployment=deployment,
            prediction_start=(
                signal_date - pd.Timedelta(days=90)
            ),
            prediction_end=signal_date,
        )

        diagnostic = model_score_diagnostic(
            baseline_prices=baseline_prices,
            challenger_prices=challenger_prices,
            signal_date=signal_date,
            deployment=deployment,
        )
        assert_model_scores_available(diagnostic)

        baseline_state = self.repository.as_paper_state(
            "Baseline_Robot"
        )
        challenger_state = self.repository.as_paper_state(
            "ML_Challenger"
        )

        dual_plan = create_dual_daily_plan(
            baseline_prices=baseline_prices,
            challenger_prices=challenger_prices,
            baseline_state=baseline_state,
            challenger_state=challenger_state,
            strategy_config=FINAL_STRATEGY_CONFIG,
            portfolio_config=FINAL_PORTFOLIO_CONFIG,
            signal_date=signal_date,
            minimum_coverage_ratio=0.60,
        )

        paths = save_dual_plans(
            dual_plan=dual_plan,
            output_root=self.settings.daily_plans_dir,
        )
        plan_directory = Path(paths["summary"]).parent
        diagnostic.to_csv(
            plan_directory / "model_diagnostic.csv",
            index=False,
        )

        marks = (
            baseline_prices.loc[
                baseline_prices["Date"].eq(signal_date),
                ["Ticker", "High", "Close"],
            ]
            .drop_duplicates("Ticker", keep="last")
        )

        for portfolio_name in (
            "Baseline_Robot",
            "ML_Challenger",
        ):
            self.repository.update_marks(
                portfolio_name,
                marks,
            )

        metadata = {
            "generated_at_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "latest_stock_date": (
                latest_stock_date.isoformat()
            ),
            "signal_date": signal_date.isoformat(),
            "data_lag_calendar_days": int(
                (latest_stock_date - signal_date).days
            ),
            "trade_ready": bool(
                latest_stock_date == signal_date
            ),
            "ticker_count": len(tickers),
            "download_error_count": int(len(errors)),
            "live_stock_file": str(
                self.settings.live_stock_path
            ),
            "live_market_file": str(
                self.settings.live_market_path
            ),
            "research_files_preserved": True,
            "model_target": deployment.target,
            "model_name": deployment.model_name,
            "filter_name": deployment.filter_name,
            "probability_threshold": (
                deployment.probability_threshold
            ),
        }

        (plan_directory / "metadata.json").write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return self.latest()
