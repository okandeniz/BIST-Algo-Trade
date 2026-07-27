"""Central project configuration.

Defaults reproduce the Robot notebook's strategy rules while keeping
the implementation configurable for later experiments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    market_symbol: str = "XU100.IS"
    start: str = "2018-01-01"
    end: str | None = None
    interval: str = "1d"
    auto_adjust: bool = True
    yfinance_repair: bool = False
    max_retries: int = 3
    pause_seconds: float = 0.20
    split_gap_threshold: float = 0.35
    split_ratio_tolerance: float = 0.20
    minimum_history_bars: int = 250


@dataclass(frozen=True)
class StrategyConfig:
    buy_score: int = 11
    watch_score: int = 8

    market_score: int = 2
    close_above_ema200_score: int = 2
    ema50_above_ema200_score: int = 2
    close_above_ema20_score: int = 1
    rsi_score: int = 1
    macd_score: int = 1
    adx_score: int = 1
    volume_score: int = 2
    breakout_score: int = 2

    minimum_rsi: float = 50.0
    minimum_adx: float = 20.0
    volume_multiplier: float = 1.30

    initial_stop_atr: float = 2.0
    trailing_stop_atr: float = 2.5
    trailing_activation_return: float = 0.06


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 500_000.0
    risk_per_trade: float = 0.0075
    max_positions: int = 8
    commission_rate: float = 0.002
    slippage_rate: float = 0.002
    minimum_valid_return: float = -0.80
    maximum_valid_return: float = 5.00
    # None: Robot'un mevcut davranışını korur.
    # 0.20: tek bir pozisyonu güncel portföy değerinin %20'siyle sınırlar.
    max_position_fraction: float | None = None
