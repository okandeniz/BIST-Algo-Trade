"""Final Robot strategy and portfolio presets.

These values were selected after development, validation, robustness,
transaction-cost and portfolio-risk analyses.
"""

from __future__ import annotations

from src.config import PortfolioConfig, StrategyConfig


FINAL_STRATEGY_CONFIG = StrategyConfig(
    buy_score=11,
    watch_score=8,
    minimum_rsi=50.0,
    minimum_adx=20.0,
    volume_multiplier=1.30,
    initial_stop_atr=1.5,
    trailing_stop_atr=2.5,
    trailing_activation_return=0.06,
)


FINAL_PORTFOLIO_CONFIG = PortfolioConfig(
    initial_capital=500_000.0,
    risk_per_trade=0.0075,
    max_positions=6,
    max_position_fraction=0.20,
    commission_rate=0.002,
    slippage_rate=0.002,
)
