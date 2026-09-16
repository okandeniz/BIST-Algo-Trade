"""Guard the product strategy names shared by the frontend and API."""

from __future__ import annotations

import unittest

from backend.app.repository import COMMISSION_RATE
from src.presets import FINAL_PORTFOLIO_CONFIG
from src.product_config import (
    PORTFOLIO_NAMES,
    PRIMARY_STRATEGY,
    STRATEGIES,
    public_trading_config,
)


class ProductConfigTests(unittest.TestCase):
    def test_baseline_is_the_primary_strategy(self) -> None:
        self.assertEqual(PRIMARY_STRATEGY, "Baseline_Robot")
        self.assertEqual(STRATEGIES[PRIMARY_STRATEGY].status, "Ana strateji")

    def test_each_strategy_has_distinct_plan_keys(self) -> None:
        buy_keys = {item.buy_plan_key for item in STRATEGIES.values()}
        sell_keys = {item.sell_plan_key for item in STRATEGIES.values()}

        self.assertEqual(len(buy_keys), len(STRATEGIES))
        self.assertEqual(len(sell_keys), len(STRATEGIES))

    def test_api_portfolio_keys_are_stable(self) -> None:
        self.assertEqual(
            set(STRATEGIES),
            {"Baseline_Robot", "RS126_Enhanced", "ML_Challenger"},
        )

    def test_promoted_portfolio_settings_have_one_source(self) -> None:
        public = public_trading_config()

        self.assertEqual(tuple(public["portfolio_names"]), PORTFOLIO_NAMES)
        self.assertEqual(
            public["portfolio"]["max_positions"],
            FINAL_PORTFOLIO_CONFIG.max_positions,
        )
        self.assertEqual(
            public["portfolio"]["commission_rate"],
            COMMISSION_RATE,
        )


if __name__ == "__main__":
    unittest.main()
