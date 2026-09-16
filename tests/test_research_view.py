"""Presentation guards for the user-facing research tables."""

from __future__ import annotations

import unittest

import pandas as pd

from frontend.views.research import _display_table


class ResearchViewTests(unittest.TestCase):
    def test_metric_table_uses_readable_labels_and_strategy_names(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "Portfolio": "Baseline_Robot",
                    "Total_Return_%": 12.3456,
                    "Max_Drawdown_%": -4.5678,
                }
            ]
        )

        result = _display_table(
            source,
            (
                "Portfolio",
                "Total_Return_%",
                "Max_Drawdown_%",
            ),
        )

        self.assertEqual(
            list(result.columns),
            ["Strateji", "Toplam getiri (%)", "Maks. düşüş (%)"],
        )
        self.assertEqual(result.loc[0, "Strateji"], "Baseline Robot")
        self.assertEqual(result.loc[0, "Toplam getiri (%)"], 12.35)


if __name__ == "__main__":
    unittest.main()
