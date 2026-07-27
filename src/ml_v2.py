"""ML V2: multi-task ranking layer for the final Robot strategy.

The primary Robot remains unchanged. ML V2 learns three complementary
targets from completed Robot trades:

1. Probability of a large winner: R_Multiple >= 2
2. Expected clipped R_Multiple
3. Probability of a stop / large-loss outcome

Predictions are converted into daily cross-sectional ranks and combined
with the original Robot rank. A small, predeclared weight/policy grid is
selected only on the 2023-2024 Validation portfolio backtest.

If a configuration passes the strict Validation acceptance rules, it is
audited using monthly expanding walk-forward training from 2025 onward.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.backtest import run_portfolio_backtest
from src.benchmark import (
    active_performance_metrics,
    align_equity_curves,
    build_benchmark_equity,
    calendar_return_table,
)
from src.config import PortfolioConfig, StrategyConfig
from src.metrics import portfolio_metrics
from src.ml_dataset import BASE_FEATURE_COLUMNS
from src.ml_training import PurgedFold


V2_EXTRA_FEATURE_COLUMNS = [
    "BREADTH_ABOVE_EMA20",
    "BREADTH_ABOVE_EMA50",
    "BREADTH_ABOVE_EMA200",
    "BREADTH_RSI50",
    "BREADTH_MACD_POSITIVE",
    "BREADTH_ADX20",
    "AL_SIGNAL_COUNT",
    "AL_SIGNAL_SHARE",
    "SCORE_CROSS_SECTION_MEAN",
    "SCORE_CROSS_SECTION_STD",
    "RET63_CROSS_SECTION_MEDIAN",
    "VOLATILITY20_CROSS_SECTION_MEDIAN",
    "RET63_RELATIVE_MEDIAN",
    "ADX_RELATIVE_MEDIAN",
    "VOLUME_RATIO_RELATIVE_MEDIAN",
    "BREAKOUT_DISTANCE_ATR",
    "TREND_ALIGNMENT_SCORE",
]

V2_FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    *V2_EXTRA_FEATURE_COLUMNS,
]


@dataclass(frozen=True)
class V2WeightProfile:
    name: str
    big_winner_weight: float
    expected_r_weight: float
    safety_weight: float
    robot_weight: float

    def validate(self) -> None:
        weights = np.array(
            [
                self.big_winner_weight,
                self.expected_r_weight,
                self.safety_weight,
                self.robot_weight,
            ],
            dtype=float,
        )

        if (weights < 0).any():
            raise ValueError(
                "V2 ağırlıkları negatif olamaz."
            )

        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(
                "V2 ağırlıklarının toplamı 1 olmalıdır."
            )


@dataclass(frozen=True)
class V2Policy:
    name: str
    top_k_per_day: int
    minimum_daily_rank_pct: float = 0.0

    def validate(self) -> None:
        if self.top_k_per_day <= 0:
            raise ValueError(
                "top_k_per_day sıfırdan büyük olmalıdır."
            )

        if not 0 <= self.minimum_daily_rank_pct <= 1:
            raise ValueError(
                "minimum_daily_rank_pct 0 ile 1 arasında olmalıdır."
            )


@dataclass(frozen=True)
class V2ModelSelection:
    big_winner_model: str
    expected_r_model: str
    stop_risk_model: str


@dataclass(frozen=True)
class V2WalkForwardConfig:
    start: str = "2025-01-01"
    end: str | None = None
    retrain_frequency: str = "MS"
    embargo_days: int = 5
    minimum_training_events: int = 500
    random_state: int = 42


DEFAULT_WEIGHT_PROFILES = [
    V2WeightProfile(
        name="Balanced",
        big_winner_weight=0.35,
        expected_r_weight=0.30,
        safety_weight=0.20,
        robot_weight=0.15,
    ),
    V2WeightProfile(
        name="Trend_Focus",
        big_winner_weight=0.45,
        expected_r_weight=0.35,
        safety_weight=0.10,
        robot_weight=0.10,
    ),
    V2WeightProfile(
        name="Risk_Adjusted",
        big_winner_weight=0.30,
        expected_r_weight=0.25,
        safety_weight=0.30,
        robot_weight=0.15,
    ),
    V2WeightProfile(
        name="Hybrid_Robot",
        big_winner_weight=0.30,
        expected_r_weight=0.25,
        safety_weight=0.20,
        robot_weight=0.25,
    ),
    V2WeightProfile(
        name="ML_Only",
        big_winner_weight=0.40,
        expected_r_weight=0.35,
        safety_weight=0.25,
        robot_weight=0.00,
    ),
]


DEFAULT_POLICIES = [
    V2Policy(
        name="Top3",
        top_k_per_day=3,
        minimum_daily_rank_pct=0.00,
    ),
    V2Policy(
        name="Top4",
        top_k_per_day=4,
        minimum_daily_rank_pct=0.00,
    ),
    V2Policy(
        name="Top6",
        top_k_per_day=6,
        minimum_daily_rank_pct=0.00,
    ),
    V2Policy(
        name="Top4_UpperHalf",
        top_k_per_day=4,
        minimum_daily_rank_pct=0.50,
    ),
    V2Policy(
        name="Top6_UpperHalf",
        top_k_per_day=6,
        minimum_daily_rank_pct=0.50,
    ),
]


def add_v2_features(
    featured_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe market breadth and relative-strength features."""
    required = {
        "Date",
        "Ticker",
        "Close",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "MACD_HIST",
        "ADX",
        "ATR_PCT",
        "VOLUME_RATIO",
        "RET_63",
        "VOLATILITY_20",
        "BREAKOUT_DISTANCE",
        "Score",
        "Signal",
    }
    missing = required.difference(
        featured_prices.columns
    )

    if missing:
        raise KeyError(
            "ML V2 özellikleri için eksik sütunlar: "
            f"{sorted(missing)}"
        )

    data = featured_prices.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    grouped = data.groupby("Date", sort=False)

    date_stats = grouped.agg(
        BREADTH_ABOVE_EMA20=(
            "EMA20",
            lambda series: np.nan,
        ),
        SCORE_CROSS_SECTION_MEAN=(
            "Score",
            "mean",
        ),
        SCORE_CROSS_SECTION_STD=(
            "Score",
            "std",
        ),
        RET63_CROSS_SECTION_MEDIAN=(
            "RET_63",
            "median",
        ),
        VOLATILITY20_CROSS_SECTION_MEDIAN=(
            "VOLATILITY_20",
            "median",
        ),
        UNIVERSE_COUNT=("Ticker", "nunique"),
        AL_SIGNAL_COUNT=(
            "Signal",
            lambda series: int(
                series.eq("AL").sum()
            ),
        ),
    ).reset_index()

    # Boolean breadth values are calculated separately because named
    # aggregation cannot reference multiple source columns cleanly.
    breadth = (
        data.assign(
            ABOVE_EMA20=data["Close"] > data["EMA20"],
            ABOVE_EMA50=data["Close"] > data["EMA50"],
            ABOVE_EMA200=data["Close"] > data["EMA200"],
            RSI50=data["RSI"] > 50,
            MACD_POSITIVE=data["MACD_HIST"] > 0,
            ADX20=data["ADX"] > 20,
        )
        .groupby("Date")
        .agg(
            BREADTH_ABOVE_EMA20=(
                "ABOVE_EMA20",
                "mean",
            ),
            BREADTH_ABOVE_EMA50=(
                "ABOVE_EMA50",
                "mean",
            ),
            BREADTH_ABOVE_EMA200=(
                "ABOVE_EMA200",
                "mean",
            ),
            BREADTH_RSI50=("RSI50", "mean"),
            BREADTH_MACD_POSITIVE=(
                "MACD_POSITIVE",
                "mean",
            ),
            BREADTH_ADX20=("ADX20", "mean"),
        )
        .reset_index()
    )

    # Remove the placeholder generated in the first named aggregation.
    date_stats = date_stats.drop(
        columns=["BREADTH_ABOVE_EMA20"]
    )

    date_stats = date_stats.merge(
        breadth,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    date_stats["AL_SIGNAL_SHARE"] = (
        date_stats["AL_SIGNAL_COUNT"]
        / date_stats["UNIVERSE_COUNT"]
    )

    data = data.merge(
        date_stats.drop(
            columns=["UNIVERSE_COUNT"]
        ),
        on="Date",
        how="left",
        validate="many_to_one",
    )

    data["RET63_RELATIVE_MEDIAN"] = (
        data["RET_63"]
        - data["RET63_CROSS_SECTION_MEDIAN"]
    )

    adx_median = grouped["ADX"].transform("median")
    volume_ratio_median = grouped[
        "VOLUME_RATIO"
    ].transform("median")

    data["ADX_RELATIVE_MEDIAN"] = (
        data["ADX"] - adx_median
    )
    data["VOLUME_RATIO_RELATIVE_MEDIAN"] = (
        data["VOLUME_RATIO"]
        - volume_ratio_median
    )

    safe_atr_pct = data["ATR_PCT"].replace(
        0,
        np.nan,
    )
    data["BREAKOUT_DISTANCE_ATR"] = (
        data["BREAKOUT_DISTANCE"]
        / safe_atr_pct
    )

    data["TREND_ALIGNMENT_SCORE"] = (
        (data["CLOSE_VS_EMA20"] > 0).astype(int)
        + (data["CLOSE_VS_EMA50"] > 0).astype(int)
        + (data["CLOSE_VS_EMA200"] > 0).astype(int)
        + (data["EMA50_VS_EMA200"] > 0).astype(int)
    )

    data[V2_FEATURE_COLUMNS] = data[
        V2_FEATURE_COLUMNS
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


def build_v2_event_dataset(
    events: pd.DataFrame,
    v2_prices: pd.DataFrame,
    expected_r_lower: float = -1.5,
    expected_r_upper: float = 5.0,
) -> pd.DataFrame:
    """Merge V2 features into trade events and create three targets."""
    required_events = {
        "Ticker",
        "Signal_Date",
        "Exit_Date",
        "R_Multiple",
        "Exit_Reason",
        "Period",
    }
    missing_events = required_events.difference(events.columns)

    if missing_events:
        raise KeyError(
            "ML V2 olay verisinde eksik sütunlar: "
            f"{sorted(missing_events)}"
        )

    required_prices = {
        "Ticker",
        "Date",
    }.union(V2_EXTRA_FEATURE_COLUMNS)
    missing_prices = required_prices.difference(
        v2_prices.columns
    )

    if missing_prices:
        raise KeyError(
            "ML V2 fiyat verisinde eksik sütunlar: "
            f"{sorted(missing_prices)}"
        )

    result = events.copy()
    result["Signal_Date"] = pd.to_datetime(
        result["Signal_Date"]
    )
    result["Exit_Date"] = pd.to_datetime(
        result["Exit_Date"]
    )

    feature_rows = (
        v2_prices[
            [
                "Ticker",
                "Date",
                *V2_EXTRA_FEATURE_COLUMNS,
            ]
        ]
        .rename(columns={"Date": "Signal_Date"})
        .drop_duplicates(
            ["Ticker", "Signal_Date"],
            keep="last",
        )
    )

    result = result.drop(
        columns=[
            column
            for column in V2_EXTRA_FEATURE_COLUMNS
            if column in result.columns
        ],
        errors="ignore",
    ).merge(
        feature_rows,
        on=["Ticker", "Signal_Date"],
        how="left",
        validate="many_to_one",
    )

    result["Big_Winner_Label_2R"] = (
        result["R_Multiple"] >= 2.0
    ).astype("int8")

    stop_reason = (
        result["Exit_Reason"]
        .astype(str)
        .str.contains(
            "Stop Loss",
            case=False,
            na=False,
        )
    )
    large_loss = result["R_Multiple"] <= -0.75

    result["Stop_Risk_Label"] = (
        stop_reason | large_loss
    ).astype("int8")

    result["Expected_R_Target"] = result[
        "R_Multiple"
    ].clip(
        lower=expected_r_lower,
        upper=expected_r_upper,
    )

    return result


def _scaled_preprocessor(
    features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _tree_preprocessor(
    features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(
                    strategy="median"
                ),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_v2_classifier_models(
    feature_columns: Iterable[str],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    features = list(feature_columns)

    return {
        "LogisticRegression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    _scaled_preprocessor(features),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.75,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "ExtraTreesClassifier": Pipeline(
            steps=[
                (
                    "preprocessor",
                    _tree_preprocessor(features),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGradientBoostingClassifier": Pipeline(
            steps=[
                (
                    "preprocessor",
                    _tree_preprocessor(features),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=350,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=1.5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def build_v2_regressor_models(
    feature_columns: Iterable[str],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    features = list(feature_columns)

    return {
        "HuberRegressor": Pipeline(
            steps=[
                (
                    "preprocessor",
                    _scaled_preprocessor(features),
                ),
                (
                    "model",
                    HuberRegressor(
                        epsilon=1.35,
                        alpha=0.001,
                        max_iter=2000,
                    ),
                ),
            ]
        ),
        "ExtraTreesRegressor": Pipeline(
            steps=[
                (
                    "preprocessor",
                    _tree_preprocessor(features),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGradientBoostingRegressor": Pipeline(
            steps=[
                (
                    "preprocessor",
                    _tree_preprocessor(features),
                ),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.04,
                        max_iter=350,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=1.5,
                        loss="squared_error",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            steps=[
                (
                    "preprocessor",
                    _tree_preprocessor(features),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _fit_classifier(
    model_name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    fitted = clone(model)

    if model_name == "HistGradientBoostingClassifier":
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


def evaluate_classifier_candidates(
    development_data: pd.DataFrame,
    folds: Iterable[PurgedFold],
    feature_columns: Iterable[str],
    target_column: str,
    models: Mapping[str, Pipeline],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate binary candidates with purged expanding folds."""
    features = list(feature_columns)
    fold_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, model in models.items():
        for fold in folds:
            train = development_data.loc[
                fold.train_indices
            ]
            validation = development_data.loc[
                fold.validation_indices
            ]

            fitted = _fit_classifier(
                model_name=model_name,
                model=model,
                X=train[features],
                y=train[target_column].astype(int),
            )

            probabilities = fitted.predict_proba(
                validation[features].replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )[:, 1]

            y_true = validation[
                target_column
            ].astype(int).to_numpy()

            base_rate = float(y_true.mean())
            pr_auc = float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )
            roc_auc = (
                float(
                    roc_auc_score(
                        y_true,
                        probabilities,
                    )
                )
                if len(np.unique(y_true)) == 2
                else np.nan
            )

            fold_records.append(
                {
                    "Task": target_column,
                    "Model": model_name,
                    "Fold": fold.fold,
                    "Base_Rate": base_rate,
                    "PR_AUC": pr_auc,
                    "PR_AUC_Lift": (
                        pr_auc - base_rate
                    ),
                    "ROC_AUC": roc_auc,
                    "Brier": float(
                        brier_score_loss(
                            y_true,
                            probabilities,
                        )
                    ),
                    "Log_Loss": float(
                        log_loss(
                            y_true,
                            probabilities,
                            labels=[0, 1],
                        )
                    ),
                    "Train_Count": len(train),
                    "Validation_Count": len(validation),
                    "Validation_Start": (
                        fold.validation_start
                    ),
                    "Validation_End": (
                        fold.validation_end
                    ),
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "Index": validation.index,
                        "Task": target_column,
                        "Model": model_name,
                        "Fold": fold.fold,
                        "Actual": y_true,
                        "Prediction": probabilities,
                    }
                )
            )

    return (
        pd.DataFrame(fold_records),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


def summarize_classifier_cv(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        fold_metrics.groupby(
            ["Task", "Model"]
        )
        .agg(
            PR_AUC_Mean=("PR_AUC", "mean"),
            PR_AUC_Min=("PR_AUC", "min"),
            PR_AUC_Lift_Mean=(
                "PR_AUC_Lift",
                "mean",
            ),
            PR_AUC_Lift_Min=(
                "PR_AUC_Lift",
                "min",
            ),
            ROC_AUC_Mean=("ROC_AUC", "mean"),
            ROC_AUC_Min=("ROC_AUC", "min"),
            Brier_Mean=("Brier", "mean"),
            Log_Loss_Mean=("Log_Loss", "mean"),
            Fold_Count=("Fold", "nunique"),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "Task",
            "PR_AUC_Lift_Mean",
            "PR_AUC_Lift_Min",
            "Brier_Mean",
        ],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def evaluate_regressor_candidates(
    development_data: pd.DataFrame,
    folds: Iterable[PurgedFold],
    feature_columns: Iterable[str],
    target_column: str,
    models: Mapping[str, Pipeline],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate expected-R models with ranking-focused metrics."""
    features = list(feature_columns)
    fold_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, model in models.items():
        for fold in folds:
            train = development_data.loc[
                fold.train_indices
            ]
            validation = development_data.loc[
                fold.validation_indices
            ]

            fitted = clone(model)
            fitted.fit(
                train[features].replace(
                    [np.inf, -np.inf],
                    np.nan,
                ),
                train[target_column],
            )

            predictions = fitted.predict(
                validation[features].replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            actual = validation[
                target_column
            ].to_numpy(dtype=float)

            correlation = spearmanr(
                actual,
                predictions,
                nan_policy="omit",
            ).statistic

            threshold = np.nanquantile(
                predictions,
                0.80,
            )
            top_mask = predictions >= threshold
            top_actual = (
                float(np.nanmean(actual[top_mask]))
                if top_mask.any()
                else np.nan
            )
            overall_actual = float(
                np.nanmean(actual)
            )

            fold_records.append(
                {
                    "Task": target_column,
                    "Model": model_name,
                    "Fold": fold.fold,
                    "MAE": float(
                        mean_absolute_error(
                            actual,
                            predictions,
                        )
                    ),
                    "RMSE": float(
                        mean_squared_error(
                            actual,
                            predictions,
                        )
                        ** 0.5
                    ),
                    "Spearman": float(correlation),
                    "Top20_Actual_R": top_actual,
                    "Overall_Actual_R": overall_actual,
                    "Top20_R_Lift": (
                        top_actual - overall_actual
                    ),
                    "Train_Count": len(train),
                    "Validation_Count": len(validation),
                    "Validation_Start": (
                        fold.validation_start
                    ),
                    "Validation_End": (
                        fold.validation_end
                    ),
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "Index": validation.index,
                        "Task": target_column,
                        "Model": model_name,
                        "Fold": fold.fold,
                        "Actual": actual,
                        "Prediction": predictions,
                    }
                )
            )

    return (
        pd.DataFrame(fold_records),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


def summarize_regressor_cv(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        fold_metrics.groupby(
            ["Task", "Model"]
        )
        .agg(
            Spearman_Mean=("Spearman", "mean"),
            Spearman_Min=("Spearman", "min"),
            Top20_R_Lift_Mean=(
                "Top20_R_Lift",
                "mean",
            ),
            Top20_R_Lift_Min=(
                "Top20_R_Lift",
                "min",
            ),
            MAE_Mean=("MAE", "mean"),
            RMSE_Mean=("RMSE", "mean"),
            Fold_Count=("Fold", "nunique"),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "Spearman_Mean",
            "Top20_R_Lift_Mean",
            "MAE_Mean",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def select_v2_models(
    classifier_summary: pd.DataFrame,
    regressor_summary: pd.DataFrame,
) -> V2ModelSelection:
    """Select one model per task using Development CV only."""
    def classifier_winner(
        target: str,
    ) -> str:
        rows = classifier_summary.loc[
            classifier_summary["Task"].eq(target)
        ]

        if rows.empty:
            raise ValueError(
                f"Classifier özeti bulunamadı: {target}"
            )

        return str(
            rows.sort_values(
                [
                    "PR_AUC_Lift_Mean",
                    "PR_AUC_Lift_Min",
                    "Brier_Mean",
                ],
                ascending=[False, False, True],
            ).iloc[0]["Model"]
        )

    if regressor_summary.empty:
        raise ValueError(
            "Regresyon CV özeti boş."
        )

    expected_r_model = str(
        regressor_summary.sort_values(
            [
                "Spearman_Mean",
                "Top20_R_Lift_Mean",
                "MAE_Mean",
            ],
            ascending=[False, False, True],
        ).iloc[0]["Model"]
    )

    return V2ModelSelection(
        big_winner_model=classifier_winner(
            "Big_Winner_Label_2R"
        ),
        expected_r_model=expected_r_model,
        stop_risk_model=classifier_winner(
            "Stop_Risk_Label"
        ),
    )


def fit_v2_models(
    training_data: pd.DataFrame,
    feature_columns: Iterable[str],
    selection: V2ModelSelection,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Fit the selected three models on one historical window."""
    features = list(feature_columns)
    classifiers = build_v2_classifier_models(
        features,
        random_state=random_state,
    )
    regressors = build_v2_regressor_models(
        features,
        random_state=random_state,
    )

    big_model = _fit_classifier(
        selection.big_winner_model,
        classifiers[selection.big_winner_model],
        training_data[features],
        training_data[
            "Big_Winner_Label_2R"
        ].astype(int),
    )
    stop_model = _fit_classifier(
        selection.stop_risk_model,
        classifiers[selection.stop_risk_model],
        training_data[features],
        training_data[
            "Stop_Risk_Label"
        ].astype(int),
    )

    expected_model = clone(
        regressors[selection.expected_r_model]
    )
    expected_model.fit(
        training_data[features].replace(
            [np.inf, -np.inf],
            np.nan,
        ),
        training_data["Expected_R_Target"],
    )

    return {
        "big_winner": big_model,
        "expected_r": expected_model,
        "stop_risk": stop_model,
    }


def score_v2_components(
    rows: pd.DataFrame,
    models: Mapping[str, Pipeline],
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Attach the three V2 model outputs to arbitrary rows."""
    features = list(feature_columns)
    result = rows.copy()

    X = result[features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result["MLV2_P_BIG_WINNER"] = (
        models["big_winner"].predict_proba(X)[:, 1]
    )
    result["MLV2_EXPECTED_R"] = (
        models["expected_r"].predict(X)
    )
    result["MLV2_P_STOP"] = (
        models["stop_risk"].predict_proba(X)[:, 1]
    )

    return result


def add_v2_composite_score(
    component_prices: pd.DataFrame,
    profile: V2WeightProfile,
) -> pd.DataFrame:
    """Convert model outputs into a daily cross-sectional composite."""
    profile.validate()
    result = component_prices.copy()

    original_al = result["Signal"].eq("AL")
    al_rows = result.loc[
        original_al,
        [
            "Date",
            "Score",
            "MLV2_P_BIG_WINNER",
            "MLV2_EXPECTED_R",
            "MLV2_P_STOP",
        ],
    ].copy()

    al_rows["MLV2_BIG_RANK"] = (
        al_rows.groupby("Date")[
            "MLV2_P_BIG_WINNER"
        ].rank(
            pct=True,
            method="average",
        )
    )
    al_rows["MLV2_EXPECTED_R_RANK"] = (
        al_rows.groupby("Date")[
            "MLV2_EXPECTED_R"
        ].rank(
            pct=True,
            method="average",
        )
    )
    al_rows["MLV2_SAFETY_RANK"] = (
        (1 - al_rows["MLV2_P_STOP"])
        .groupby(al_rows["Date"])
        .rank(
            pct=True,
            method="average",
        )
    )
    al_rows["MLV2_ROBOT_RANK"] = (
        al_rows.groupby("Date")["Score"]
        .rank(
            pct=True,
            method="average",
        )
    )

    al_rows["MLV2_COMPOSITE"] = (
        profile.big_winner_weight
        * al_rows["MLV2_BIG_RANK"]
        + profile.expected_r_weight
        * al_rows["MLV2_EXPECTED_R_RANK"]
        + profile.safety_weight
        * al_rows["MLV2_SAFETY_RANK"]
        + profile.robot_weight
        * al_rows["MLV2_ROBOT_RANK"]
    )

    al_rows["MLV2_DAILY_RANK_PCT"] = (
        al_rows.groupby("Date")[
            "MLV2_COMPOSITE"
        ].rank(
            pct=True,
            method="first",
        )
    )
    al_rows["MLV2_DAILY_RANK"] = (
        al_rows.groupby("Date")[
            "MLV2_COMPOSITE"
        ].rank(
            ascending=False,
            method="first",
        )
    )

    columns_to_add = [
        "MLV2_BIG_RANK",
        "MLV2_EXPECTED_R_RANK",
        "MLV2_SAFETY_RANK",
        "MLV2_ROBOT_RANK",
        "MLV2_COMPOSITE",
        "MLV2_DAILY_RANK_PCT",
        "MLV2_DAILY_RANK",
    ]

    for column in columns_to_add:
        result[column] = np.nan
        result.loc[
            al_rows.index,
            column,
        ] = al_rows[column]

    result["MLV2_WEIGHT_PROFILE"] = profile.name
    return result


def apply_v2_policy(
    composite_prices: pd.DataFrame,
    policy: V2Policy,
) -> pd.DataFrame:
    """Filter and rank Robot AL rows using the frozen V2 policy."""
    policy.validate()
    result = composite_prices.copy()
    original_al = result["Signal"].eq("AL")

    pass_mask = (
        original_al
        & result["MLV2_DAILY_RANK"].le(
            policy.top_k_per_day
        )
        & result["MLV2_DAILY_RANK_PCT"].ge(
            policy.minimum_daily_rank_pct
        )
        & result["MLV2_COMPOSITE"].notna()
    )

    result["Robot_Score_Original"] = result["Score"]
    result.loc[
        original_al & ~pass_mask,
        "Signal",
    ] = "ALMA"

    # The event-driven engine ranks by Score first. Assign a synthetic
    # integer score to accepted rows so their execution priority follows
    # the ML V2 composite rank while keeping the original score audited.
    accepted_rank = result.loc[
        pass_mask,
        "MLV2_DAILY_RANK",
    ].astype(int)

    result.loc[
        pass_mask,
        "Score",
    ] = 1_000 - accepted_rank

    result["MLV2_FILTER_PASSED"] = False
    result.loc[
        pass_mask,
        "MLV2_FILTER_PASSED",
    ] = True
    result["MLV2_POLICY"] = policy.name

    return result


def _portfolio_result(
    name: str,
    prices: pd.DataFrame,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    period = prices.loc[
        pd.to_datetime(prices["Date"]).between(
            pd.Timestamp(start),
            pd.Timestamp(end),
            inclusive="both",
        )
    ].copy()

    equity, trades = run_portfolio_backtest(
        period,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
    )
    metrics = portfolio_metrics(
        equity,
        trades,
    )
    metrics["Configuration"] = name
    return metrics, equity, trades


def run_v2_validation_grid(
    component_prices: pd.DataFrame,
    weight_profiles: Iterable[V2WeightProfile],
    policies: Iterable[V2Policy],
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, pd.DataFrame]],
]:
    """Run the predeclared V2 weight and policy grid on Validation."""
    baseline_metrics, baseline_equity, baseline_trades = (
        _portfolio_result(
            "Baseline_Robot",
            component_prices,
            strategy_config,
            portfolio_config,
            start,
            end,
        )
    )

    original_al_count = int(
        component_prices.loc[
            pd.to_datetime(
                component_prices["Date"]
            ).between(
                pd.Timestamp(start),
                pd.Timestamp(end),
                inclusive="both",
            ),
            "Signal",
        ].eq("AL").sum()
    )

    records = [
        {
            **baseline_metrics,
            "Weight_Profile": None,
            "Policy": None,
            "Passed_AL_Rows": original_al_count,
            "Signal_Pass_Rate_%": 100.0,
        }
    ]
    outputs = {
        "Baseline_Robot": {
            "equity": baseline_equity,
            "trades": baseline_trades,
        }
    }

    for profile in weight_profiles:
        composite = add_v2_composite_score(
            component_prices,
            profile,
        )

        for policy in policies:
            filtered = apply_v2_policy(
                composite,
                policy,
            )
            name = f"{profile.name}__{policy.name}"

            metrics, equity, trades = (
                _portfolio_result(
                    name,
                    filtered,
                    strategy_config,
                    portfolio_config,
                    start,
                    end,
                )
            )

            passed = int(
                filtered.loc[
                    pd.to_datetime(
                        filtered["Date"]
                    ).between(
                        pd.Timestamp(start),
                        pd.Timestamp(end),
                        inclusive="both",
                    ),
                    "MLV2_FILTER_PASSED",
                ].sum()
            )

            records.append(
                {
                    **metrics,
                    "Weight_Profile": profile.name,
                    "Policy": policy.name,
                    "Passed_AL_Rows": passed,
                    "Signal_Pass_Rate_%": (
                        passed
                        / original_al_count
                        * 100
                        if original_al_count > 0
                        else np.nan
                    ),
                }
            )
            outputs[name] = {
                "equity": equity,
                "trades": trades,
            }

    return pd.DataFrame(records), outputs


def v2_acceptance_table(
    validation_results: pd.DataFrame,
    minimum_cagr_improvement_pp: float = 3.0,
    minimum_trade_fraction: float = 0.50,
    maximum_drawdown_deterioration_pp: float = 2.0,
) -> pd.DataFrame:
    """Apply the predeclared hurdle required to challenge Baseline."""
    rows = validation_results.copy()
    baseline_rows = rows.loc[
        rows["Configuration"].eq(
            "Baseline_Robot"
        )
    ]

    if len(baseline_rows) != 1:
        raise ValueError(
            "Validation tablosunda tek Baseline gerekir."
        )

    baseline = baseline_rows.iloc[0]

    rows["CAGR_Difference_pp"] = (
        rows["CAGR_%"] - baseline["CAGR_%"]
    )
    rows["Max_DD_Difference_pp"] = (
        rows["Max_Drawdown_%"]
        - baseline["Max_Drawdown_%"]
    )
    rows["Profit_Factor_Difference"] = (
        rows["Profit_Factor"]
        - baseline["Profit_Factor"]
    )
    rows["Sharpe_Difference"] = (
        rows["Sharpe"] - baseline["Sharpe"]
    )
    rows["Calmar_Difference"] = (
        rows["Calmar"] - baseline["Calmar"]
    )

    rows["Enough_Trades"] = (
        rows["Trade_Count"]
        >= baseline["Trade_Count"]
        * minimum_trade_fraction
    )
    rows["CAGR_Hurdle_Passed"] = (
        rows["CAGR_Difference_pp"]
        >= minimum_cagr_improvement_pp
    )
    rows["Profit_Factor_Not_Worse"] = (
        rows["Profit_Factor"]
        >= baseline["Profit_Factor"]
    )
    rows["Sharpe_Not_Worse"] = (
        rows["Sharpe"] >= baseline["Sharpe"]
    )
    rows["Drawdown_Acceptable"] = (
        rows["Max_Drawdown_%"]
        >= (
            baseline["Max_Drawdown_%"]
            - maximum_drawdown_deterioration_pp
        )
    )

    rows["MLV2_Accepted"] = (
        ~rows["Configuration"].eq(
            "Baseline_Robot"
        )
        & rows["Enough_Trades"]
        & rows["CAGR_Hurdle_Passed"]
        & rows["Profit_Factor_Not_Worse"]
        & rows["Sharpe_Not_Worse"]
        & rows["Drawdown_Acceptable"]
    )

    rows["Acceptance_Count"] = rows[
        [
            "Enough_Trades",
            "CAGR_Hurdle_Passed",
            "Profit_Factor_Not_Worse",
            "Sharpe_Not_Worse",
            "Drawdown_Acceptable",
        ]
    ].sum(axis=1)

    return rows.sort_values(
        [
            "MLV2_Accepted",
            "Acceptance_Count",
            "CAGR_Difference_pp",
            "Calmar",
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


def select_v2_validation_champion(
    acceptance_table: pd.DataFrame,
) -> tuple[dict[str, Any] | None, bool]:
    accepted = acceptance_table.loc[
        acceptance_table["MLV2_Accepted"]
    ]

    if accepted.empty:
        return None, False

    champion = (
        accepted.sort_values(
            [
                "CAGR_Difference_pp",
                "Calmar",
                "Profit_Factor",
                "Sharpe",
            ],
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )
    return champion, True


def _walk_forward_blocks(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    frequency: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    scheduled = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq=frequency,
    )

    starts = [start_ts]
    starts.extend(
        date
        for date in scheduled
        if date > start_ts
    )
    starts = sorted(set(starts))

    blocks = []
    for index, block_start in enumerate(starts):
        if index + 1 < len(starts):
            block_end = (
                starts[index + 1]
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


def generate_v2_walk_forward_components(
    v2_prices: pd.DataFrame,
    event_data: pd.DataFrame,
    selection: V2ModelSelection,
    feature_columns: Iterable[str],
    config: V2WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate monthly multi-task predictions without future labels."""
    features = list(feature_columns)
    prices = v2_prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    for column in (
        "MLV2_P_BIG_WINNER",
        "MLV2_EXPECTED_R",
        "MLV2_P_STOP",
    ):
        prices[column] = np.nan

    events = event_data.copy()
    events["Signal_Date"] = pd.to_datetime(
        events["Signal_Date"]
    )
    events["Exit_Date"] = pd.to_datetime(
        events["Exit_Date"]
    )

    end = (
        pd.Timestamp(config.end)
        if config.end is not None
        else pd.Timestamp(prices["Date"].max())
    )

    logs: list[dict[str, Any]] = []

    for block_number, (
        block_start,
        block_end,
    ) in enumerate(
        _walk_forward_blocks(
            config.start,
            end,
            config.retrain_frequency,
        ),
        start=1,
    ):
        purge_boundary = (
            block_start
            - pd.Timedelta(
                days=config.embargo_days
            )
        )

        training = events.loc[
            (events["Signal_Date"] < block_start)
            & (events["Exit_Date"] < purge_boundary)
        ].copy()

        if len(training) < config.minimum_training_events:
            raise ValueError(
                f"{block_start.date()} için eğitim "
                f"olayı yetersiz: {len(training)}"
            )

        models = fit_v2_models(
            training,
            features,
            selection,
            random_state=config.random_state,
        )

        block_mask = prices["Date"].between(
            block_start,
            block_end,
            inclusive="both",
        )
        prediction_mask = (
            block_mask
            & prices["Signal"].eq("AL")
        )

        scored_count = int(
            prediction_mask.sum()
        )
        imputed_rows = 0
        missing_cells = 0

        if scored_count:
            X = prices.loc[
                prediction_mask,
                features,
            ].replace(
                [np.inf, -np.inf],
                np.nan,
            )
            imputed_rows = int(
                X.isna().any(axis=1).sum()
            )
            missing_cells = int(
                X.isna().sum().sum()
            )

            scored = score_v2_components(
                prices.loc[
                    prediction_mask
                ].copy(),
                models,
                features,
            )

            for column in (
                "MLV2_P_BIG_WINNER",
                "MLV2_EXPECTED_R",
                "MLV2_P_STOP",
            ):
                prices.loc[
                    prediction_mask,
                    column,
                ] = scored[column].to_numpy()

        logs.append(
            {
                "Block": block_number,
                "Block_Start": block_start,
                "Block_End": block_end,
                "Purge_Boundary": purge_boundary,
                "Training_Event_Count": len(training),
                "Training_Signal_End": (
                    training["Signal_Date"].max()
                ),
                "Training_Exit_End": (
                    training["Exit_Date"].max()
                ),
                "Big_Winner_Rate_%": (
                    training[
                        "Big_Winner_Label_2R"
                    ].mean()
                    * 100
                ),
                "Stop_Risk_Rate_%": (
                    training[
                        "Stop_Risk_Label"
                    ].mean()
                    * 100
                ),
                "Scored_AL_Rows": scored_count,
                "Rows_With_Imputation": imputed_rows,
                "Missing_Feature_Cells": missing_cells,
                "Rows_With_Imputation_%": (
                    imputed_rows
                    / scored_count
                    * 100
                    if scored_count > 0
                    else 0.0
                ),
                "Mean_P_BIG_WINNER": (
                    prices.loc[
                        prediction_mask,
                        "MLV2_P_BIG_WINNER",
                    ].mean()
                ),
                "Mean_EXPECTED_R": (
                    prices.loc[
                        prediction_mask,
                        "MLV2_EXPECTED_R",
                    ].mean()
                ),
                "Mean_P_STOP": (
                    prices.loc[
                        prediction_mask,
                        "MLV2_P_STOP",
                    ].mean()
                ),
            }
        )

    period_al = (
        prices["Date"].between(
            pd.Timestamp(config.start),
            end,
            inclusive="both",
        )
        & prices["Signal"].eq("AL")
    )

    missing_predictions = prices.loc[
        period_al,
        [
            "MLV2_P_BIG_WINNER",
            "MLV2_EXPECTED_R",
            "MLV2_P_STOP",
        ],
    ].isna().any(axis=1)

    if missing_predictions.any():
        raise RuntimeError(
            "Walk-forward ML V2 tahminlerinde "
            "skorsuz AL satırı kaldı."
        )

    return prices, pd.DataFrame(logs)


def evaluate_v2_walk_forward(
    component_prices: pd.DataFrame,
    profile: V2WeightProfile,
    policy: V2Policy,
    market_prices: pd.DataFrame,
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
    """Compare Baseline, frozen ML V2 policy and BIST100."""
    composite = add_v2_composite_score(
        component_prices,
        profile,
    )
    filtered = apply_v2_policy(
        composite,
        policy,
    )

    baseline_metrics, baseline_equity, baseline_trades = (
        _portfolio_result(
            "Baseline_Robot",
            component_prices,
            strategy_config,
            portfolio_config,
            start,
            end,
        )
    )
    v2_metrics, v2_equity, v2_trades = (
        _portfolio_result(
            "ML_V2_Challenger",
            filtered,
            strategy_config,
            portfolio_config,
            start,
            end,
        )
    )

    period_original_al = component_prices.loc[
        pd.to_datetime(
            component_prices["Date"]
        ).between(
            pd.Timestamp(start),
            pd.Timestamp(end),
            inclusive="both",
        ),
        "Signal",
    ].eq("AL").sum()

    period_passed = filtered.loc[
        pd.to_datetime(
            filtered["Date"]
        ).between(
            pd.Timestamp(start),
            pd.Timestamp(end),
            inclusive="both",
        ),
        "MLV2_FILTER_PASSED",
    ].sum()

    baseline_metrics["Signal_Pass_Rate_%"] = 100.0
    v2_metrics["Signal_Pass_Rate_%"] = (
        period_passed
        / period_original_al
        * 100
        if period_original_al > 0
        else np.nan
    )

    benchmark = build_benchmark_equity(
        market_prices=market_prices,
        comparison_dates=baseline_equity["Date"],
        initial_capital=(
            portfolio_config.initial_capital
        ),
        include_costs=False,
        benchmark_name="BIST100",
    )
    benchmark_metrics = portfolio_metrics(
        benchmark,
        pd.DataFrame(columns=["Return"]),
    )
    benchmark_metrics["Configuration"] = (
        "BIST100_Gross"
    )
    benchmark_metrics[
        "Signal_Pass_Rate_%"
    ] = np.nan

    metrics = pd.DataFrame(
        [
            baseline_metrics,
            v2_metrics,
            benchmark_metrics,
        ]
    )

    equity_comparison = (
        baseline_equity[
            ["Date", "Equity"]
        ]
        .rename(
            columns={
                "Equity": "Baseline_Robot"
            }
        )
        .merge(
            v2_equity[
                ["Date", "Equity"]
            ].rename(
                columns={
                    "Equity": "ML_V2_Challenger"
                }
            ),
            on="Date",
            how="inner",
        )
        .merge(
            benchmark[
                ["Date", "Equity"]
            ].rename(
                columns={
                    "Equity": "BIST100_Gross"
                }
            ),
            on="Date",
            how="inner",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    yearly = calendar_return_table(
        {
            "Baseline_Robot": baseline_equity,
            "ML_V2_Challenger": v2_equity,
            "BIST100_Gross": benchmark,
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
    for name, equity in (
        ("Baseline_Robot", baseline_equity),
        ("ML_V2_Challenger", v2_equity),
    ):
        active = active_performance_metrics(
            align_equity_curves(
                equity,
                benchmark,
            )
        )
        active["Portfolio"] = name
        active_records.append(active)

    outputs = {
        "Baseline_Robot": {
            "equity": baseline_equity,
            "trades": baseline_trades,
        },
        "ML_V2_Challenger": {
            "equity": v2_equity,
            "trades": v2_trades,
        },
        "BIST100_Gross": {
            "equity": benchmark,
            "trades": pd.DataFrame(),
        },
    }

    return (
        metrics,
        equity_comparison,
        yearly_table,
        pd.DataFrame(active_records),
        outputs,
    )


def save_v2_artifacts(
    output_directory: str | Path,
    classifier_cv: pd.DataFrame,
    regressor_cv: pd.DataFrame,
    model_selection: V2ModelSelection,
    validation_grid: pd.DataFrame,
    acceptance_table: pd.DataFrame,
    champion: dict[str, Any] | None,
    walk_forward_metrics: pd.DataFrame | None = None,
    walk_forward_equity: pd.DataFrame | None = None,
    walk_forward_yearly: pd.DataFrame | None = None,
    walk_forward_active: pd.DataFrame | None = None,
    walk_forward_log: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_directory)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "classifier_cv": (
            output_dir
            / "v2_classifier_cv_summary.csv"
        ),
        "regressor_cv": (
            output_dir
            / "v2_regressor_cv_summary.csv"
        ),
        "validation_grid": (
            output_dir
            / "v2_validation_portfolio_grid.csv"
        ),
        "acceptance": (
            output_dir
            / "v2_validation_acceptance.csv"
        ),
        "decision": (
            output_dir
            / "v2_decision.json"
        ),
    }

    classifier_cv.to_csv(
        paths["classifier_cv"],
        index=False,
    )
    regressor_cv.to_csv(
        paths["regressor_cv"],
        index=False,
    )
    validation_grid.to_csv(
        paths["validation_grid"],
        index=False,
    )
    acceptance_table.to_csv(
        paths["acceptance"],
        index=False,
    )

    decision = {
        "model_selection": asdict(model_selection),
        "validation_champion": champion,
        "validation_accepted": champion is not None,
    }

    with paths["decision"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            decision,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    optional_frames = {
        "walk_forward_metrics": (
            walk_forward_metrics,
            "v2_walk_forward_metrics.csv",
            "csv",
        ),
        "walk_forward_equity": (
            walk_forward_equity,
            "v2_walk_forward_equity.parquet",
            "parquet",
        ),
        "walk_forward_yearly": (
            walk_forward_yearly,
            "v2_walk_forward_yearly.csv",
            "csv",
        ),
        "walk_forward_active": (
            walk_forward_active,
            "v2_walk_forward_active.csv",
            "csv",
        ),
        "walk_forward_log": (
            walk_forward_log,
            "v2_walk_forward_training_log.csv",
            "csv",
        ),
    }

    for key, (
        frame,
        filename,
        file_type,
    ) in optional_frames.items():
        if frame is None:
            continue

        path = output_dir / filename
        if file_type == "parquet":
            frame.to_parquet(
                path,
                index=False,
            )
        else:
            frame.to_csv(
                path,
                index=False,
            )
        paths[key] = path

    return paths
