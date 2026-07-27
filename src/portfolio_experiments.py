"""Portfolio-level parameter experiments for the Robot strategy."""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from typing import Iterable, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm

from src.config import PortfolioConfig, StrategyConfig
from src.experiments import evaluate_strategy


def portfolio_parameter_combinations(
    parameter_grid: Mapping[str, Sequence[object]],
) -> list[dict[str, object]]:
    """Convert a PortfolioConfig grid into explicit dictionaries."""
    names = list(parameter_grid)
    value_lists = [parameter_grid[name] for name in names]

    return [
        dict(zip(names, values, strict=True))
        for values in product(*value_lists)
    ]


def run_portfolio_grid(
    stock_features: pd.DataFrame,
    market_regime: pd.Series,
    strategy_config: StrategyConfig,
    base_portfolio: PortfolioConfig,
    parameter_grid: Mapping[str, Sequence[object]],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Evaluate PortfolioConfig combinations on one date period."""
    combinations = portfolio_parameter_combinations(parameter_grid)
    records: list[dict[str, object]] = []

    iterator: Iterable[dict[str, object]] = combinations

    if show_progress:
        iterator = tqdm(
            combinations,
            desc=f"Portföy grid: {start} → {end}",
        )

    for experiment_id, parameters in enumerate(iterator, start=1):
        portfolio_config = replace(
            base_portfolio,
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

            metrics.update(
                {
                    "Experiment_ID": experiment_id,
                    "Status": "OK",
                    "Error": "",
                }
            )

        except Exception as error:
            metrics = {
                "Experiment_ID": experiment_id,
                "Status": "ERROR",
                "Error": str(error),
                "Period_Start": pd.Timestamp(start),
                "Period_End": pd.Timestamp(end),
            }

            for key, value in asdict(portfolio_config).items():
                metrics[f"Portfolio_{key}"] = value

        records.append(metrics)

    return pd.DataFrame(records)


def _restore_none_parameter(
    parameter_name: str,
    value: object,
) -> object:
    """Restore None after pandas converts optional values to NaN."""
    if (
        parameter_name == "max_position_fraction"
        and pd.isna(value)
    ):
        return None

    return value


def compare_portfolio_periods(
    stock_features: pd.DataFrame,
    market_regime: pd.Series,
    configurations: pd.DataFrame,
    strategy_config: StrategyConfig,
    base_portfolio: PortfolioConfig,
    periods: Mapping[str, tuple[str, str]],
    parameter_columns: Sequence[str],
) -> pd.DataFrame:
    """Evaluate selected portfolio configurations across named periods."""
    records: list[dict[str, object]] = []

    for config_number, (_, row) in enumerate(
        configurations.iterrows(),
        start=1,
    ):
        parameters = {}

        for column in parameter_columns:
            parameter_name = column.removeprefix("Portfolio_")
            parameters[parameter_name] = _restore_none_parameter(
                parameter_name,
                row[column],
            )

        portfolio_config = replace(
            base_portfolio,
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
