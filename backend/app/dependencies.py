"""Cached application dependencies."""

from __future__ import annotations

from functools import lru_cache

from backend.app.config import (
    AppSettings,
    get_settings,
)
from backend.app.database import Database
from backend.app.repository import TradingRepository
from backend.app.services.backtest_service import (
    BacktestService,
)
from backend.app.services.market_service import MarketService
from backend.app.services.signal_service import SignalService
from backend.app.services.walkforward_service import WalkForwardService


@lru_cache(maxsize=1)
def get_database() -> Database:
    settings = get_settings()
    return Database(settings.database_path)


@lru_cache(maxsize=1)
def get_repository() -> TradingRepository:
    return TradingRepository(get_database())


@lru_cache(maxsize=1)
def get_market_service() -> MarketService:
    settings = get_settings()
    return MarketService(
        live_stock_path=settings.live_stock_path,
        fallback_stock_path=settings.processed_stock_path,
    )


@lru_cache(maxsize=1)
def get_backtest_service() -> BacktestService:
    settings = get_settings()
    return BacktestService(
        settings.backtest_results_dir
    )


@lru_cache(maxsize=1)
def get_signal_service() -> SignalService:
    return SignalService(
        settings=get_settings(),
        repository=get_repository(),
    )



@lru_cache(maxsize=1)
def get_walkforward_service() -> WalkForwardService:
    settings = get_settings()

    return WalkForwardService(
        settings.project_root
        / "results"
        / "ml"
        / "walk_forward"
    )
