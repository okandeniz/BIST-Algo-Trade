"""Promoted product configuration shared by API, UI and persistence.

Research modules may define experimental variants, but the running product must
import portfolio identities and promoted trading parameters from this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.presets import FINAL_PORTFOLIO_CONFIG, FINAL_STRATEGY_CONFIG


BASELINE_PORTFOLIO = "Baseline_Robot"
ENHANCED_PORTFOLIO = "RS126_Enhanced"
ML_PORTFOLIO = "ML_Challenger"
PRIMARY_STRATEGY = BASELINE_PORTFOLIO


@dataclass(frozen=True)
class StrategyInfo:
    key: str
    label: str
    status: str
    buy_plan_key: str
    sell_plan_key: str
    description: str


STRATEGIES = {
    BASELINE_PORTFOLIO: StrategyInfo(
        key=BASELINE_PORTFOLIO,
        label="Baseline Robot",
        status="Ana strateji",
        buy_plan_key="baseline_buys",
        sell_plan_key="baseline_sells",
        description="Günlük kullanım için varsayılan, kilitli Robot kuralları.",
    ),
    ENHANCED_PORTFOLIO: StrategyInfo(
        key=ENHANCED_PORTFOLIO,
        label="RS126 Enhanced",
        status="Challenger",
        buy_plan_key="enhanced_buys",
        sell_plan_key="enhanced_sells",
        description=(
            "RS126 zayıf olduğunda Robot skorunu bir puan azaltan "
            "deneysel varyant."
        ),
    ),
    ML_PORTFOLIO: StrategyInfo(
        key=ML_PORTFOLIO,
        label="ML Challenger",
        status="Challenger",
        buy_plan_key="challenger_buys",
        sell_plan_key="challenger_sells",
        description=(
            "Robot adaylarını kilitli makine öğrenmesi filtresiyle süzen "
            "deneysel varyant."
        ),
    ),
}

PORTFOLIO_NAMES = tuple(STRATEGIES)


def strategy_options() -> list[str]:
    return list(PORTFOLIO_NAMES)


def strategy_label(key: str) -> str:
    strategy = STRATEGIES[key]
    return f"{strategy.label} · {strategy.status}"


def public_trading_config() -> dict:
    """Serializable promoted settings safe to expose through the API."""
    return {
        "primary_strategy": PRIMARY_STRATEGY,
        "portfolio_names": list(PORTFOLIO_NAMES),
        "strategy": asdict(FINAL_STRATEGY_CONFIG),
        "portfolio": asdict(FINAL_PORTFOLIO_CONFIG),
    }
