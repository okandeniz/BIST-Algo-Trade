"""RS126 Enhanced Baseline paper-trading helpers.

The existing Baseline Robot remains unchanged. This module creates a separate
challenger whose only additional rule is:

    RS126 <= 0  -> Robot score - 1 point

Missing RS126 values are not penalized. The rule primarily removes marginal
score-11 candidates while preserving stronger Baseline signals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import StrategyConfig
from src.daily_signal import DailyPlan
from src.product_config import ENHANCED_PORTFOLIO


PORTFOLIO_NAME = ENHANCED_PORTFOLIO
RULE_NAME = "RS126_Negative_Penalty_1"
RULE_DESCRIPTION = "RS126 <= 0 ise Robot skorundan 1 puan düş"


def build_rs126_enhanced_prices(
    baseline_prices: pd.DataFrame,
    market_features: pd.DataFrame,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    """Create enhanced scores without mutating the Baseline price frame."""
    required_stock = {
        "Date",
        "Ticker",
        "RET_126",
        "Score",
        "Signal",
    }
    required_market = {
        "Date",
        "RET_126",
    }

    missing_stock = required_stock.difference(
        baseline_prices.columns
    )
    missing_market = required_market.difference(
        market_features.columns
    )

    if missing_stock:
        raise KeyError(
            "Enhanced Baseline için hisse verisinde eksik "
            f"sütunlar: {sorted(missing_stock)}"
        )
    if missing_market:
        raise KeyError(
            "Enhanced Baseline için endeks verisinde eksik "
            f"sütunlar: {sorted(missing_market)}"
        )

    result = baseline_prices.copy()
    result["Date"] = pd.to_datetime(result["Date"])

    market_returns = (
        market_features[["Date", "RET_126"]]
        .copy()
        .assign(
            Date=lambda frame: pd.to_datetime(
                frame["Date"]
            )
        )
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .rename(
            columns={
                "RET_126": "MARKET_RET_126",
            }
        )
    )

    # Avoid duplicate columns when the function is called on an already
    # enriched frame during a notebook or API rerun.
    for column in (
        "MARKET_RET_126",
        "RS_126",
        "Baseline_Score",
        "RS126_Penalty",
        "Enhanced_Score",
        "Enhanced_Rule",
    ):
        if column in result.columns:
            result = result.drop(columns=column)

    result = result.merge(
        market_returns,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    result["RS_126"] = (
        pd.to_numeric(
            result["RET_126"],
            errors="coerce",
        )
        - pd.to_numeric(
            result["MARKET_RET_126"],
            errors="coerce",
        )
    )

    result["Baseline_Score"] = pd.to_numeric(
        result["Score"],
        errors="coerce",
    )

    valid_rs126 = result["RS_126"].notna()
    penalty_mask = (
        valid_rs126
        & result["RS_126"].le(0.0)
    )

    result["RS126_Penalty"] = (
        penalty_mask.astype(int)
    )
    result["Enhanced_Score"] = (
        result["Baseline_Score"]
        - result["RS126_Penalty"]
    )

    result["Score"] = result[
        "Enhanced_Score"
    ]

    result["Signal"] = np.select(
        [
            result["Enhanced_Score"].ge(
                strategy_config.buy_score
            ),
            result["Enhanced_Score"].ge(
                strategy_config.watch_score
            ),
        ],
        [
            "AL",
            "İZLE",
        ],
        default="ALMA",
    )

    result["Enhanced_Rule"] = np.where(
        penalty_mask,
        RULE_DESCRIPTION,
        "Ceza uygulanmadı",
    )

    if "Reasons" in result.columns:
        reasons = (
            result["Reasons"]
            .fillna("")
            .astype(str)
        )

        addition = (
            " | RS126<=0: -1 puan"
        )

        result["Reasons"] = np.where(
            penalty_mask,
            reasons.str.rstrip(" |") + addition,
            reasons,
        )

    return result.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)


def attach_enhanced_diagnostics(
    plan: DailyPlan,
    enhanced_prices: pd.DataFrame,
) -> DailyPlan:
    """Attach RS126 diagnostics to the enhanced buy-order table."""
    if plan.buy_orders.empty:
        return plan

    latest_rows = (
        enhanced_prices.loc[
            pd.to_datetime(
                enhanced_prices["Date"]
            ).eq(plan.signal_date),
            [
                "Ticker",
                "Baseline_Score",
                "Enhanced_Score",
                "RS_126",
                "RS126_Penalty",
            ],
        ]
        .drop_duplicates(
            "Ticker",
            keep="last",
        )
    )

    plan.buy_orders = plan.buy_orders.merge(
        latest_rows,
        on="Ticker",
        how="left",
        validate="many_to_one",
    )

    return plan


def _read_csv_safely(
    path: Path,
) -> pd.DataFrame:
    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _enhanced_buy_comparison(
    plan: DailyPlan,
) -> pd.DataFrame:
    columns = [
        "Portfolio",
        "Ticker",
        "Rank",
        "Score",
        "Estimated_Entry",
        "Estimated_Stop",
        "Estimated_Shares",
        "ML_Probability",
        "Baseline_Score",
        "Enhanced_Score",
        "RS_126",
        "RS126_Penalty",
    ]

    if plan.buy_orders.empty:
        return pd.DataFrame(
            columns=columns
        )

    frame = plan.buy_orders.copy()
    frame["Portfolio"] = PORTFOLIO_NAME
    frame["ML_Probability"] = np.nan

    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan

    return frame[columns]


def _enhanced_sell_comparison(
    plan: DailyPlan,
) -> pd.DataFrame:
    columns = [
        "Portfolio",
        "Ticker",
        "Action",
        "Reason",
        "Close",
        "Stop_Loss",
        "Trailing_Level",
    ]

    if plan.sell_orders.empty:
        return pd.DataFrame(
            columns=columns
        )

    frame = plan.sell_orders.copy()
    frame["Portfolio"] = PORTFOLIO_NAME

    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan

    return frame[columns]


def save_rs126_enhanced_plan(
    plan: DailyPlan,
    plan_directory: str | Path,
) -> dict[str, Path]:
    """Persist the enhanced plan and append it to comparison tables."""
    directory = Path(plan_directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "summary": (
            directory
            / "enhanced_summary.csv"
        ),
        "buys": (
            directory
            / "enhanced_buy_orders.csv"
        ),
        "sells": (
            directory
            / "enhanced_sell_orders.csv"
        ),
        "holds": (
            directory
            / "enhanced_hold_positions.csv"
        ),
        "candidates": (
            directory
            / "enhanced_ranked_candidates.csv"
        ),
    }

    enhanced_summary = (
        plan.summary.copy().assign(
            Portfolio=PORTFOLIO_NAME,
            Strategy_Rule=RULE_NAME,
        )
    )

    enhanced_summary.to_csv(
        paths["summary"],
        index=False,
    )
    plan.buy_orders.to_csv(
        paths["buys"],
        index=False,
    )
    plan.sell_orders.to_csv(
        paths["sells"],
        index=False,
    )
    plan.hold_positions.to_csv(
        paths["holds"],
        index=False,
    )
    plan.ranked_candidates.to_csv(
        paths["candidates"],
        index=False,
    )

    # Add the enhanced portfolio to the existing synchronized comparison
    # tables without changing the established Baseline/ML plan filenames.
    summary_path = (
        directory / "dual_summary.csv"
    )
    summary = _read_csv_safely(
        summary_path
    )
    summary = summary.loc[
        ~summary.get(
            "Portfolio",
            pd.Series(
                dtype=str,
                index=summary.index,
            ),
        ).eq(PORTFOLIO_NAME)
    ].copy()

    pd.concat(
        [
            summary,
            enhanced_summary,
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(
        summary_path,
        index=False,
    )

    buy_path = (
        directory
        / "dual_buy_comparison.csv"
    )
    buy_comparison = _read_csv_safely(
        buy_path
    )

    if (
        not buy_comparison.empty
        and "Portfolio" in buy_comparison.columns
    ):
        buy_comparison = buy_comparison.loc[
            ~buy_comparison[
                "Portfolio"
            ].eq(PORTFOLIO_NAME)
        ].copy()

    buy_comparison = pd.concat(
        [
            buy_comparison,
            _enhanced_buy_comparison(plan),
        ],
        ignore_index=True,
        sort=False,
    )

    if (
        not buy_comparison.empty
        and {"Ticker", "Portfolio"}.issubset(
            buy_comparison.columns
        )
    ):
        counts = (
            buy_comparison.groupby(
                "Ticker"
            )["Portfolio"]
            .nunique()
            .rename("Portfolio_Count")
        )

        buy_comparison = (
            buy_comparison.drop(
                columns=[
                    "Portfolio_Count",
                    "Appears_In_Both",
                    "Appears_In_All_Three",
                ],
                errors="ignore",
            )
            .merge(
                counts,
                on="Ticker",
                how="left",
            )
        )
        buy_comparison[
            "Appears_In_Multiple"
        ] = buy_comparison[
            "Portfolio_Count"
        ].ge(2)
        buy_comparison[
            "Appears_In_All_Three"
        ] = buy_comparison[
            "Portfolio_Count"
        ].eq(3)

    buy_comparison.to_csv(
        buy_path,
        index=False,
    )

    sell_path = (
        directory
        / "dual_sell_comparison.csv"
    )
    sell_comparison = _read_csv_safely(
        sell_path
    )

    if (
        not sell_comparison.empty
        and "Portfolio" in sell_comparison.columns
    ):
        sell_comparison = sell_comparison.loc[
            ~sell_comparison[
                "Portfolio"
            ].eq(PORTFOLIO_NAME)
        ].copy()

    pd.concat(
        [
            sell_comparison,
            _enhanced_sell_comparison(plan),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(
        sell_path,
        index=False,
    )

    return paths
