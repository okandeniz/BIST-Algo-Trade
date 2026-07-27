"""Controlled parameter experiments for the Robot strategy.

The module evaluates configurations on explicit date windows and keeps
parameter selection separate from the untouched holdout period.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from typing import Iterable, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm

from src.backtest import run_portfolio_backtest
from src.config import PortfolioConfig, StrategyConfig
from src.metrics import portfolio_metrics
from src.signals import add_robot_scores


def slice_period(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return an inclusive date slice."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    result = df.loc[
        df["Date"].between(start_ts, end_ts, inclusive="both")
    ].copy()

    return result.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def evaluate_strategy(
    stock_features: pd.DataFrame,
    market_regime: pd.Series,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Score prices, run one backtest and return metrics plus outputs."""
    scored = add_robot_scores(
        stock_features=stock_features,
        market_regime=market_regime,
        config=strategy_config,
        include_reasons=False,
    )

    period_prices = slice_period(scored, start=start, end=end)

    if period_prices.empty:
        raise ValueError(
            f"Seçilen dönemde veri yok: {start} - {end}"
        )

    equity_df, trades_df = run_portfolio_backtest(
        scored_prices=period_prices,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
    )

    metrics = portfolio_metrics(equity_df, trades_df)

    metrics.update({
        "Period_Start": pd.Timestamp(start),
        "Period_End": pd.Timestamp(end),
    })

    for key, value in asdict(strategy_config).items():
        metrics[f"Strategy_{key}"] = value

    for key, value in asdict(portfolio_config).items():
        metrics[f"Portfolio_{key}"] = value

    return metrics, equity_df, trades_df


def parameter_combinations(
    parameter_grid: Mapping[str, Sequence[object]],
) -> list[dict[str, object]]:
    """Convert a parameter grid into explicit dictionaries."""
    names = list(parameter_grid)
    values = [parameter_grid[name] for name in names]

    return [
        dict(zip(names, combination, strict=True))
        for combination in product(*values)
    ]


def run_strategy_grid(
    stock_features: pd.DataFrame,
    market_regime: pd.Series,
    base_strategy: StrategyConfig,
    portfolio_config: PortfolioConfig,
    parameter_grid: Mapping[str, Sequence[object]],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Evaluate every StrategyConfig combination on one date window."""
    combinations = parameter_combinations(parameter_grid)
    records: list[dict[str, object]] = []

    iterator: Iterable[dict[str, object]] = combinations

    if show_progress:
        iterator = tqdm(
            combinations,
            desc=f"Strateji grid: {start} → {end}",
        )

    for experiment_id, parameters in enumerate(iterator, start=1):
        strategy_config = replace(
            base_strategy,
            **parameters,
        )

        try:
            metrics, _, _ = evaluate_strategy(
                stock_features=stock_features,
                market_regime=market_regime,
                strategy_config=strategy_config,
                portfolio_config=portfolio_config,
                start=start,
                end=end,
            )

            metrics["Experiment_ID"] = experiment_id
            metrics["Status"] = "OK"
            metrics["Error"] = ""

        except Exception as error:
            metrics = {
                "Experiment_ID": experiment_id,
                "Status": "ERROR",
                "Error": str(error),
                "Period_Start": pd.Timestamp(start),
                "Period_End": pd.Timestamp(end),
            }

            for key, value in asdict(strategy_config).items():
                metrics[f"Strategy_{key}"] = value

        records.append(metrics)

    return pd.DataFrame(records)


def apply_robustness_filters(
    results: pd.DataFrame,
    minimum_trades: int = 40,
    maximum_drawdown_limit: float = -35.0,
    minimum_profit_factor: float = 1.10,
) -> pd.DataFrame:
    """Keep configurations that meet basic robustness constraints.

    maximum_drawdown_limit=-35 means drawdowns below -35% are rejected.
    """
    required = {
        "Status",
        "Trade_Count",
        "Max_Drawdown_%",
        "Profit_Factor",
        "Calmar",
        "CAGR_%",
    }

    missing = required.difference(results.columns)
    if missing:
        raise KeyError(
            f"Sonuç tablosunda eksik metrikler: {sorted(missing)}"
        )

    filtered = results.loc[
        results["Status"].eq("OK")
        & results["Trade_Count"].ge(minimum_trades)
        & results["Max_Drawdown_%"].ge(maximum_drawdown_limit)
        & results["Profit_Factor"].ge(minimum_profit_factor)
    ].copy()

    return filtered.sort_values(
        ["Calmar", "CAGR_%", "Profit_Factor", "Sharpe"],
        ascending=False,
    ).reset_index(drop=True)


def compare_periods(
    stock_features: pd.DataFrame,
    market_regime: pd.Series,
    configurations: pd.DataFrame,
    base_strategy: StrategyConfig,
    portfolio_config: PortfolioConfig,
    periods: Mapping[str, tuple[str, str]],
    parameter_columns: Sequence[str],
) -> pd.DataFrame:
    """Evaluate selected parameter rows across development/validation periods."""
    records: list[dict[str, object]] = []

    for config_number, (_, row) in enumerate(
        configurations.iterrows(),
        start=1,
    ):
        parameters = {
            column.removeprefix("Strategy_"): row[column]
            for column in parameter_columns
        }

        strategy_config = replace(
            base_strategy,
            **parameters,
        )

        for period_name, (start, end) in periods.items():
            metrics, _, _ = evaluate_strategy(
                stock_features=stock_features,
                market_regime=market_regime,
                strategy_config=strategy_config,
                portfolio_config=portfolio_config,
                start=start,
                end=end,
            )

            metrics["Selected_Config"] = config_number
            metrics["Period_Name"] = period_name
            records.append(metrics)

    return pd.DataFrame(records)
