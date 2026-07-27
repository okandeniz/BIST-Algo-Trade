"""Leakage-controlled walk-forward ML Challenger comparison.

Methodological boundary
-----------------------
The target/model/filter combination was selected using data through
2024. Therefore the honest comparison begins on 2025-01-01.

At the beginning of every calendar month:

1. Use only signal events strictly before the month.
2. Purge events whose exit date is not at least `embargo_days` before
   the new month.
3. Refit the locked model type on all eligible completed events.
4. Score only Robot AL rows in that month.
5. Apply the locked Validation threshold without retuning it.

No 2025+ outcome is used before it becomes observable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.benchmark import (
    active_performance_metrics,
    align_equity_curves,
    build_benchmark_equity,
    calendar_return_table,
)
from src.config import PortfolioConfig, StrategyConfig
from src.metrics import portfolio_metrics
from src.ml_portfolio import (
    MLFilterConfig,
    run_filter_grid,
)
from src.ml_training import build_candidate_models


@dataclass(frozen=True)
class WalkForwardConfig:
    start: str = "2025-01-01"
    end: str | None = None
    retrain_frequency: str = "MS"
    embargo_days: int = 5
    minimum_training_events: int = 500
    random_state: int = 42


def _fit_candidate(
    model_name: str,
    model_template: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    fitted = clone(model_template)

    if model_name == "HistGradientBoosting":
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y,
        )
        fitted.fit(
            X,
            y,
            model__sample_weight=sample_weight,
        )
    else:
        fitted.fit(X, y)

    return fitted


def _walk_forward_blocks(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    frequency: str = "MS",
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    if start_ts > end_ts:
        raise ValueError(
            "Walk-forward başlangıcı bitişten sonra olamaz."
        )

    scheduled = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq=frequency,
    )

    block_starts = [start_ts]
    block_starts.extend(
        timestamp
        for timestamp in scheduled
        if timestamp > start_ts
    )
    block_starts = sorted(set(block_starts))

    blocks: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for index, block_start in enumerate(block_starts):
        if index + 1 < len(block_starts):
            block_end = (
                block_starts[index + 1]
                - pd.Timedelta(days=1)
            )
        else:
            block_end = end_ts

        blocks.append(
            (
                pd.Timestamp(block_start),
                min(pd.Timestamp(block_end), end_ts),
            )
        )

    return blocks


def generate_walk_forward_probabilities(
    featured_prices: pd.DataFrame,
    events: pd.DataFrame,
    model_name: str,
    target_column: str,
    feature_columns: Iterable[str],
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate monthly expanding-window probabilities without leakage."""
    features = list(feature_columns)

    required_prices = {
        "Date",
        "Ticker",
        "Signal",
    }.union(features)
    missing_prices = required_prices.difference(
        featured_prices.columns
    )

    if missing_prices:
        raise KeyError(
            "Walk-forward fiyat verisinde eksik sütunlar: "
            f"{sorted(missing_prices)}"
        )

    required_events = {
        "Signal_Date",
        "Exit_Date",
        target_column,
    }.union(features)
    missing_events = required_events.difference(events.columns)

    if missing_events:
        raise KeyError(
            "Walk-forward olay verisinde eksik sütunlar: "
            f"{sorted(missing_events)}"
        )

    prices = featured_prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["ML_Probability"] = np.nan
    prices["WF_Training_End"] = pd.NaT
    prices["WF_Training_Event_Count"] = np.nan
    prices["WF_Block_Start"] = pd.NaT
    prices["WF_Block_End"] = pd.NaT

    event_data = events.copy()
    event_data["Signal_Date"] = pd.to_datetime(
        event_data["Signal_Date"]
    )
    event_data["Exit_Date"] = pd.to_datetime(
        event_data["Exit_Date"]
    )

    end = (
        pd.Timestamp(config.end)
        if config.end is not None
        else pd.Timestamp(prices["Date"].max())
    )

    candidate_models = build_candidate_models(
        feature_columns=features,
        random_state=config.random_state,
    )

    if model_name not in candidate_models:
        raise KeyError(
            f"Desteklenmeyen model: {model_name}"
        )

    model_template = candidate_models[model_name]
    logs: list[dict[str, Any]] = []

    for block_number, (
        block_start,
        block_end,
    ) in enumerate(
        _walk_forward_blocks(
            start=config.start,
            end=end,
            frequency=config.retrain_frequency,
        ),
        start=1,
    ):
        purge_boundary = (
            block_start
            - pd.Timedelta(
                days=config.embargo_days
            )
        )

        training = event_data.loc[
            (event_data["Signal_Date"] < block_start)
            & (event_data["Exit_Date"] < purge_boundary)
        ].copy()

        training = training.loc[
            training[target_column].notna()
        ].copy()

        # Candidate pipelines already include SimpleImputer. Missing and
        # infinite values are handled using statistics learned only from
        # this historical training window.
        training_features = (
            training[features]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        if len(training) < config.minimum_training_events:
            raise ValueError(
                f"{block_start.date()} bloğu için eğitim "
                f"olayı yetersiz: {len(training)}"
            )

        if training[target_column].nunique() < 2:
            raise ValueError(
                f"{block_start.date()} eğitim verisinde "
                "iki sınıf da bulunmuyor."
            )

        fitted_model = _fit_candidate(
            model_name=model_name,
            model_template=model_template,
            X=training_features,
            y=training[target_column].astype(int),
        )

        block_mask = prices["Date"].between(
            block_start,
            block_end,
            inclusive="both",
        )
        robot_al_mask = prices["Signal"].eq("AL")

        # Score every Robot AL row. The fitted pipeline's SimpleImputer
        # fills missing features using this block's historical training
        # statistics, so no future information is used.
        prediction_mask = (
            block_mask
            & robot_al_mask
        )

        probabilities = np.array([], dtype=float)
        imputed_row_count = 0
        missing_feature_cells = 0

        if prediction_mask.any():
            prediction_features = (
                prices.loc[
                    prediction_mask,
                    features,
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            imputed_row_count = int(
                prediction_features.isna().any(
                    axis=1
                ).sum()
            )
            missing_feature_cells = int(
                prediction_features.isna().sum().sum()
            )

            probabilities = fitted_model.predict_proba(
                prediction_features
            )[:, 1]

            prices.loc[
                prediction_mask,
                "ML_Probability",
            ] = probabilities
            prices.loc[
                prediction_mask,
                "WF_Training_End",
            ] = training["Exit_Date"].max()
            prices.loc[
                prediction_mask,
                "WF_Training_Event_Count",
            ] = len(training)
            prices.loc[
                prediction_mask,
                "WF_Block_Start",
            ] = block_start
            prices.loc[
                prediction_mask,
                "WF_Block_End",
            ] = block_end

        all_al_count = int(
            (
                block_mask
                & robot_al_mask
            ).sum()
        )
        scored_count = int(prediction_mask.sum())

        logs.append(
            {
                "Block": block_number,
                "Block_Start": block_start,
                "Block_End": block_end,
                "Purge_Boundary": purge_boundary,
                "Training_Event_Count": len(training),
                "Training_Signal_Start": (
                    training["Signal_Date"].min()
                ),
                "Training_Signal_End": (
                    training["Signal_Date"].max()
                ),
                "Training_Exit_End": (
                    training["Exit_Date"].max()
                ),
                "Training_Positive_Rate_%": (
                    training[target_column].mean() * 100
                ),
                "Robot_AL_Rows": all_al_count,
                "Scored_AL_Rows": scored_count,
                "Score_Coverage_%": (
                    scored_count / all_al_count * 100
                    if all_al_count > 0
                    else 100.0
                ),
                "Rows_With_Imputation": (
                    imputed_row_count
                ),
                "Missing_Feature_Cells": (
                    missing_feature_cells
                ),
                "Rows_With_Imputation_%": (
                    imputed_row_count
                    / scored_count
                    * 100
                    if scored_count > 0
                    else 0.0
                ),
                "Probability_Mean": (
                    float(probabilities.mean())
                    if len(probabilities)
                    else np.nan
                ),
                "Probability_Std": (
                    float(probabilities.std(ddof=1))
                    if len(probabilities) > 1
                    else np.nan
                ),
            }
        )

    training_log = pd.DataFrame(logs)

    period_mask = prices["Date"].between(
        pd.Timestamp(config.start),
        end,
        inclusive="both",
    )
    al_mask = period_mask & prices["Signal"].eq("AL")
    missing_score_mask = (
        al_mask
        & prices["ML_Probability"].isna()
    )

    if missing_score_mask.any():
        missing_rows = prices.loc[
            missing_score_mask,
            ["Date", "Ticker"],
        ].head(20)

        raise RuntimeError(
            "Walk-forward model tahmini sonrasında hâlâ "
            "skorsuz Robot AL satırı bulundu. Bu artık özellik "
            "eksikliğinden değil, model tahmin akışındaki teknik "
            "bir sorundan kaynaklanır. İlk örnekler: "
            + missing_rows.astype(str).to_dict(
                orient="records"
            ).__str__()
        )

    return prices, training_log


def evaluate_walk_forward_comparison(
    probability_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    probability_threshold: float,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, pd.DataFrame]],
]:
    """Compare Baseline Robot, ML Challenger and BIST100."""
    filters = [
        MLFilterConfig(
            name="Baseline_Robot",
        ),
        MLFilterConfig(
            name="ML_Challenger",
            probability_threshold=(
                probability_threshold
            ),
        ),
    ]

    strategy_results, outputs = run_filter_grid(
        probability_prices=probability_prices,
        filters=filters,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
        start=start,
        end=end,
    )

    baseline_equity = outputs[
        "Baseline_Robot"
    ]["equity"]

    benchmark_equity = build_benchmark_equity(
        market_prices=market_prices,
        comparison_dates=baseline_equity["Date"],
        initial_capital=(
            portfolio_config.initial_capital
        ),
        include_costs=False,
        benchmark_name="BIST100",
    )

    empty_trades = pd.DataFrame(
        columns=["Return"]
    )
    benchmark_metrics = portfolio_metrics(
        benchmark_equity,
        empty_trades,
    )

    metrics_records = []

    for _, row in strategy_results.iterrows():
        metrics_records.append(
            {
                "Portfolio": row["Filter_Name"],
                **{
                    column: row[column]
                    for column in (
                        "Start_Value",
                        "End_Value",
                        "Total_Return_%",
                        "CAGR_%",
                        "Max_Drawdown_%",
                        "Profit_Factor",
                        "Win_Rate_%",
                        "Expectancy_%",
                        "Sharpe",
                        "Sortino",
                        "Calmar",
                        "Trade_Count",
                        "Exposure_%",
                        "Signal_Pass_Rate_%",
                    )
                    if column in row
                },
            }
        )

    metrics_records.append(
        {
            "Portfolio": "BIST100_Gross",
            **benchmark_metrics,
            "Signal_Pass_Rate_%": np.nan,
        }
    )

    comparison_metrics = pd.DataFrame(
        metrics_records
    )

    baseline_curve = (
        outputs["Baseline_Robot"]["equity"][
            ["Date", "Equity"]
        ]
        .rename(
            columns={
                "Equity": "Baseline_Robot"
            }
        )
    )
    challenger_curve = (
        outputs["ML_Challenger"]["equity"][
            ["Date", "Equity"]
        ]
        .rename(
            columns={
                "Equity": "ML_Challenger"
            }
        )
    )
    benchmark_curve = (
        benchmark_equity[["Date", "Equity"]]
        .rename(
            columns={
                "Equity": "BIST100_Gross"
            }
        )
    )

    equity_comparison = (
        baseline_curve.merge(
            challenger_curve,
            on="Date",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            benchmark_curve,
            on="Date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    yearly = calendar_return_table(
        {
            "Baseline_Robot": (
                outputs["Baseline_Robot"]["equity"]
            ),
            "ML_Challenger": (
                outputs["ML_Challenger"]["equity"]
            ),
            "BIST100_Gross": benchmark_equity,
        },
        frequency="YE",
    )
    yearly["Year"] = yearly["Date"].dt.year
    yearly_table = yearly.pivot_table(
        index="Year",
        columns="Portfolio",
        values="Return_%",
        aggfunc="first",
    ).reset_index()

    active_records = []

    for portfolio_name in (
        "Baseline_Robot",
        "ML_Challenger",
    ):
        aligned = align_equity_curves(
            outputs[portfolio_name]["equity"],
            benchmark_equity,
        )
        active = active_performance_metrics(
            aligned
        )
        active["Portfolio"] = portfolio_name
        active_records.append(active)

    active_table = pd.DataFrame(
        active_records
    )

    return (
        comparison_metrics,
        equity_comparison,
        yearly_table,
        active_table,
        outputs,
    )


def save_walk_forward_artifacts(
    output_directory: str | Path,
    comparison_metrics: pd.DataFrame,
    equity_comparison: pd.DataFrame,
    yearly_table: pd.DataFrame,
    active_table: pd.DataFrame,
    training_log: pd.DataFrame,
    probability_prices: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    output_dir = Path(output_directory)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "metrics": (
            output_dir
            / "walk_forward_comparison_metrics.csv"
        ),
        "equity": (
            output_dir
            / "walk_forward_equity.parquet"
        ),
        "yearly": (
            output_dir
            / "walk_forward_yearly.csv"
        ),
        "active": (
            output_dir
            / "walk_forward_active_metrics.csv"
        ),
        "training_log": (
            output_dir
            / "walk_forward_training_log.csv"
        ),
        "probabilities": (
            output_dir
            / "walk_forward_probabilities.parquet"
        ),
        "metadata": (
            output_dir
            / "walk_forward_metadata.json"
        ),
    }

    comparison_metrics.to_csv(
        paths["metrics"],
        index=False,
    )
    equity_comparison.to_parquet(
        paths["equity"],
        index=False,
    )
    yearly_table.to_csv(
        paths["yearly"],
        index=False,
    )
    active_table.to_csv(
        paths["active"],
        index=False,
    )
    training_log.to_csv(
        paths["training_log"],
        index=False,
    )

    probability_columns = [
        column
        for column in (
            "Date",
            "Ticker",
            "Signal",
            "ML_Probability",
            "WF_Training_End",
            "WF_Training_Event_Count",
            "WF_Block_Start",
            "WF_Block_End",
        )
        if column in probability_prices.columns
    ]
    probability_prices[
        probability_columns
    ].to_parquet(
        paths["probabilities"],
        index=False,
    )

    with paths["metadata"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return paths
