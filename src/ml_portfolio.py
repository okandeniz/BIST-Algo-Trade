"""Portfolio-level evaluation of Robot meta-label probabilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import run_portfolio_backtest
from src.config import PortfolioConfig, StrategyConfig
from src.metrics import portfolio_metrics


@dataclass(frozen=True)
class MLFilterConfig:
    """One probability-based filter applied only to Robot AL rows."""

    name: str
    probability_threshold: float | None = None
    keep_top_fraction: float | None = None

    def validate(self) -> None:
        supplied = sum(
            value is not None
            for value in (
                self.probability_threshold,
                self.keep_top_fraction,
            )
        )

        if supplied > 1:
            raise ValueError(
                "Aynı filtrede threshold ve keep_top_fraction "
                "birlikte kullanılamaz."
            )

        if (
            self.probability_threshold is not None
            and not 0 <= self.probability_threshold <= 1
        ):
            raise ValueError(
                "probability_threshold 0 ile 1 arasında olmalıdır."
            )

        if (
            self.keep_top_fraction is not None
            and not 0 < self.keep_top_fraction <= 1
        ):
            raise ValueError(
                "keep_top_fraction 0 ile 1 arasında olmalıdır."
            )


def add_model_probabilities(
    featured_prices: pd.DataFrame,
    fitted_model: Any,
    feature_columns: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    probability_column: str = "ML_Probability",
) -> pd.DataFrame:
    """Predict every eligible Robot AL row in an explicit date window."""
    features = list(feature_columns)
    required = {
        "Date",
        "Ticker",
        "Signal",
    }.union(features)

    missing = required.difference(featured_prices.columns)

    if missing:
        raise KeyError(
            f"ML tahmini için eksik sütunlar: {sorted(missing)}"
        )

    result = featured_prices.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    result[probability_column] = np.nan

    period_mask = result["Date"].between(
        pd.Timestamp(start),
        pd.Timestamp(end),
        inclusive="both",
    )
    al_mask = result["Signal"].eq("AL")
    complete_mask = result[features].notna().all(axis=1)

    prediction_mask = (
        period_mask
        & al_mask
        & complete_mask
    )

    if prediction_mask.any():
        probabilities = fitted_model.predict_proba(
            result.loc[prediction_mask, features]
        )[:, 1]

        result.loc[
            prediction_mask,
            probability_column,
        ] = probabilities

    return result


def apply_ml_filter(
    probability_prices: pd.DataFrame,
    filter_config: MLFilterConfig,
    probability_column: str = "ML_Probability",
) -> pd.DataFrame:
    """Turn rejected Robot AL rows into ALMA without changing exits."""
    filter_config.validate()

    required = {
        "Date",
        "Ticker",
        "Signal",
        probability_column,
    }
    missing = required.difference(
        probability_prices.columns
    )

    if missing:
        raise KeyError(
            f"ML filtresi için eksik sütunlar: {sorted(missing)}"
        )

    result = probability_prices.copy()
    original_al = result["Signal"].eq("AL")

    if (
        filter_config.probability_threshold is None
        and filter_config.keep_top_fraction is None
    ):
        pass_mask = original_al.copy()
        result["ML_Daily_Rank_Pct"] = np.nan

    elif filter_config.probability_threshold is not None:
        pass_mask = (
            original_al
            & result[probability_column].ge(
                filter_config.probability_threshold
            )
        )
        result["ML_Daily_Rank_Pct"] = np.nan

    else:
        al_probabilities = result.loc[
            original_al,
            ["Date", probability_column],
        ].copy()

        rank_values = (
            al_probabilities.groupby("Date")[
                probability_column
            ]
            .rank(
                pct=True,
                method="average",
                ascending=True,
            )
        )

        result["ML_Daily_Rank_Pct"] = np.nan
        result.loc[
            al_probabilities.index,
            "ML_Daily_Rank_Pct",
        ] = rank_values

        cutoff = 1 - float(
            filter_config.keep_top_fraction
        )

        pass_mask = (
            original_al
            & result["ML_Daily_Rank_Pct"].ge(cutoff)
            & result[probability_column].notna()
        )

    rejected_mask = original_al & ~pass_mask
    result.loc[rejected_mask, "Signal"] = "ALMA"

    result["ML_Filter_Passed"] = False
    result.loc[pass_mask, "ML_Filter_Passed"] = True
    result["ML_Filter_Name"] = filter_config.name

    return result


def evaluate_filter(
    probability_prices: pd.DataFrame,
    filter_config: MLFilterConfig,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Backtest one ML filter inside the original event-driven engine."""
    period = probability_prices.loc[
        pd.to_datetime(
            probability_prices["Date"]
        ).between(
            pd.Timestamp(start),
            pd.Timestamp(end),
            inclusive="both",
        )
    ].copy()

    if period.empty:
        raise ValueError(
            f"Seçilen dönemde veri yok: {start} - {end}"
        )

    filtered = apply_ml_filter(
        period,
        filter_config=filter_config,
    )

    equity, trades = run_portfolio_backtest(
        scored_prices=filtered,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
    )

    metrics = portfolio_metrics(
        equity,
        trades,
    )

    original_al = period["Signal"].eq("AL")
    passed_al = filtered["ML_Filter_Passed"]

    metrics.update(
        {
            "Filter_Name": filter_config.name,
            "Probability_Threshold": (
                filter_config.probability_threshold
            ),
            "Keep_Top_Fraction": (
                filter_config.keep_top_fraction
            ),
            "Original_AL_Rows": int(
                original_al.sum()
            ),
            "Passed_AL_Rows": int(
                passed_al.sum()
            ),
            "Signal_Pass_Rate_%": (
                passed_al.sum()
                / original_al.sum()
                * 100
                if original_al.sum() > 0
                else np.nan
            ),
            "Period_Start": pd.Timestamp(start),
            "Period_End": pd.Timestamp(end),
        }
    )

    return metrics, equity, trades


def run_filter_grid(
    probability_prices: pd.DataFrame,
    filters: Iterable[MLFilterConfig],
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, pd.DataFrame]],
]:
    """Evaluate baseline plus all supplied ML filters."""
    records = []
    outputs: dict[
        str,
        dict[str, pd.DataFrame],
    ] = {}

    for filter_config in filters:
        metrics, equity, trades = evaluate_filter(
            probability_prices=probability_prices,
            filter_config=filter_config,
            strategy_config=strategy_config,
            portfolio_config=portfolio_config,
            start=start,
            end=end,
        )

        records.append(metrics)
        outputs[filter_config.name] = {
            "equity": equity,
            "trades": trades,
        }

    return pd.DataFrame(records), outputs


def add_baseline_differences(
    results: pd.DataFrame,
    baseline_name: str = "Baseline_Robot",
) -> pd.DataFrame:
    """Add metric differences relative to the unfiltered Robot."""
    baseline_rows = results.loc[
        results["Filter_Name"].eq(
            baseline_name
        )
    ]

    if len(baseline_rows) != 1:
        raise ValueError(
            "Karşılaştırma tablosunda tek bir baseline satırı olmalıdır."
        )

    baseline = baseline_rows.iloc[0]
    output = results.copy()

    metric_columns = [
        "CAGR_%",
        "Max_Drawdown_%",
        "Profit_Factor",
        "Sharpe",
        "Calmar",
        "Trade_Count",
    ]

    for metric in metric_columns:
        output[f"{metric}_Difference"] = (
            output[metric] - baseline[metric]
        )

    return output


def validation_acceptance_table(
    validation_results: pd.DataFrame,
    baseline_name: str = "Baseline_Robot",
    minimum_trade_fraction: float = 0.50,
    maximum_drawdown_deterioration_pp: float = 3.0,
) -> pd.DataFrame:
    """Apply strict portfolio-level acceptance criteria to ML filters."""
    if not 0 < minimum_trade_fraction <= 1:
        raise ValueError(
            "minimum_trade_fraction 0 ile 1 arasında olmalıdır."
        )

    rows = validation_results.copy()
    baseline = rows.loc[
        rows["Filter_Name"].eq(baseline_name)
    ]

    if len(baseline) != 1:
        raise ValueError(
            "Acceptance için tek bir baseline satırı gerekir."
        )

    base = baseline.iloc[0]

    rows["Enough_Trades"] = (
        rows["Trade_Count"]
        >= base["Trade_Count"]
        * minimum_trade_fraction
    )
    rows["CAGR_Improved"] = (
        rows["CAGR_%"] > base["CAGR_%"]
    )
    rows["Calmar_Improved"] = (
        rows["Calmar"] > base["Calmar"]
    )
    rows["Profit_Factor_Not_Worse"] = (
        rows["Profit_Factor"]
        >= base["Profit_Factor"]
    )
    rows["Drawdown_Acceptable"] = (
        rows["Max_Drawdown_%"]
        >= (
            base["Max_Drawdown_%"]
            - maximum_drawdown_deterioration_pp
        )
    )

    rows["ML_Accepted"] = (
        ~rows["Filter_Name"].eq(baseline_name)
        & rows["Enough_Trades"]
        & rows["CAGR_Improved"]
        & rows["Calmar_Improved"]
        & rows["Profit_Factor_Not_Worse"]
        & rows["Drawdown_Acceptable"]
    )

    rows["Acceptance_Count"] = rows[
        [
            "Enough_Trades",
            "CAGR_Improved",
            "Calmar_Improved",
            "Profit_Factor_Not_Worse",
            "Drawdown_Acceptable",
        ]
    ].sum(axis=1)

    return rows.sort_values(
        [
            "ML_Accepted",
            "Acceptance_Count",
            "Calmar",
            "CAGR_%",
            "Profit_Factor",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)


def select_validation_champion(
    acceptance_table: pd.DataFrame,
    baseline_name: str = "Baseline_Robot",
) -> tuple[str, bool]:
    """Select an accepted ML filter or retain the original Robot."""
    accepted = acceptance_table.loc[
        acceptance_table["ML_Accepted"]
    ]

    if accepted.empty:
        return baseline_name, False

    champion = (
        accepted.sort_values(
            [
                "Calmar",
                "CAGR_%",
                "Profit_Factor",
                "Sharpe",
            ],
            ascending=False,
        )
        .iloc[0]["Filter_Name"]
    )

    return str(champion), True
