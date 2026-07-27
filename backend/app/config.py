"""Application settings for the FastAPI and Streamlit integration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _default_project_root() -> Path:
    configured = os.getenv("PROJECT_ROOT")

    if configured:
        return Path(configured).expanduser().resolve()

    # backend/app/config.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    database_path: Path
    processed_stock_path: Path
    live_stock_path: Path
    live_market_path: Path
    ticker_file: Path
    backtest_results_dir: Path
    daily_plans_dir: Path
    app_results_dir: Path
    signal_lookback_days: int
    cors_origins: tuple[str, ...]


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    project_root = _default_project_root()

    database_path = Path(
        os.getenv(
            "APP_DATABASE_PATH",
            str(
                project_root
                / "results"
                / "app"
                / "trading.db"
            ),
        )
    ).expanduser().resolve()

    cors_origins = tuple(
        value.strip()
        for value in os.getenv(
            "CORS_ORIGINS",
            (
                "http://localhost:8501,"
                "http://127.0.0.1:8501"
            ),
        ).split(",")
        if value.strip()
    )

    return AppSettings(
        project_root=project_root,
        database_path=database_path,
        processed_stock_path=(
            project_root
            / "data"
            / "processed"
            / "bist100_robot_clean.parquet"
        ),
        live_stock_path=(
            project_root
            / "data"
            / "live"
            / "bist100_live_clean.parquet"
        ),
        live_market_path=(
            project_root
            / "data"
            / "live"
            / "xu100_live_clean.parquet"
        ),
        ticker_file=(
            project_root
            / "data"
            / "raw"
            / "bist100_sirketler.xlsx"
        ),
        backtest_results_dir=project_root / "results",
        daily_plans_dir=(
            project_root
            / "results"
            / "paper_trading"
            / "dual"
            / "daily_plans"
        ),
        app_results_dir=(
            project_root
            / "results"
            / "app"
        ),
        signal_lookback_days=int(
            os.getenv("SIGNAL_LOOKBACK_DAYS", "900")
        ),
        cors_origins=cors_origins,
    )
