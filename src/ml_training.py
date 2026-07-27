"""Time-aware model training utilities for Robot meta-labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import json
import numpy as np
import pandas as pd

from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


@dataclass(frozen=True)
class PurgedFold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_signal_end: pd.Timestamp
    train_exit_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def build_candidate_models(
    feature_columns: Iterable[str],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Create fixed, interpretable baseline candidate models."""
    features = list(feature_columns)

    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
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

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=12,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def create_purged_expanding_folds(
    development_data: pd.DataFrame,
    n_splits: int = 4,
    initial_train_fraction: float = 0.40,
    embargo_days: int = 5,
) -> list[PurgedFold]:
    """Create expanding date folds and purge labels crossing validation."""
    if not 0 < initial_train_fraction < 1:
        raise ValueError(
            "initial_train_fraction 0 ile 1 arasında olmalıdır."
        )

    if n_splits < 2:
        raise ValueError("n_splits en az 2 olmalıdır.")

    data = development_data.copy()
    data["Signal_Date"] = pd.to_datetime(data["Signal_Date"])
    data["Exit_Date"] = pd.to_datetime(data["Exit_Date"])

    unique_dates = np.array(
        sorted(data["Signal_Date"].drop_duplicates())
    )

    if len(unique_dates) < 50:
        raise ValueError(
            "Purged CV için yeterli benzersiz sinyal tarihi yok."
        )

    initial_position = max(
        1,
        int(len(unique_dates) * initial_train_fraction),
    )

    validation_dates = unique_dates[initial_position:]
    validation_blocks = np.array_split(
        validation_dates,
        n_splits,
    )

    folds: list[PurgedFold] = []

    for fold_number, block in enumerate(
        validation_blocks,
        start=1,
    ):
        if len(block) == 0:
            continue

        validation_start = pd.Timestamp(block[0])
        validation_end = pd.Timestamp(block[-1])
        purge_boundary = (
            validation_start
            - pd.Timedelta(days=embargo_days)
        )

        train_mask = (
            (data["Signal_Date"] < validation_start)
            & (data["Exit_Date"] < purge_boundary)
        )

        validation_mask = data["Signal_Date"].between(
            validation_start,
            validation_end,
            inclusive="both",
        )

        train_indices = data.index[train_mask].to_numpy()
        validation_indices = data.index[
            validation_mask
        ].to_numpy()

        if len(train_indices) == 0 or len(validation_indices) == 0:
            continue

        train_rows = data.loc[train_indices]

        folds.append(
            PurgedFold(
                fold=fold_number,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_signal_end=pd.Timestamp(
                    train_rows["Signal_Date"].max()
                ),
                train_exit_end=pd.Timestamp(
                    train_rows["Exit_Date"].max()
                ),
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )

    if len(folds) < 2:
        raise ValueError(
            "Yeterli sayıda geçerli purged fold üretilemedi."
        )

    return folds


def folds_summary(
    data: pd.DataFrame,
    folds: Iterable[PurgedFold],
    target_column: str = "Meta_Label",
) -> pd.DataFrame:
    """Return an auditable summary of each purged fold."""
    records = []

    for fold in folds:
        train = data.loc[fold.train_indices]
        validation = data.loc[fold.validation_indices]

        records.append(
            {
                "Fold": fold.fold,
                "Train_Count": len(train),
                "Validation_Count": len(validation),
                "Train_Positive_Rate_%": (
                    train[target_column].mean() * 100
                ),
                "Validation_Positive_Rate_%": (
                    validation[target_column].mean() * 100
                ),
                "Train_Signal_End": fold.train_signal_end,
                "Train_Exit_End": fold.train_exit_end,
                "Validation_Start": fold.validation_start,
                "Validation_End": fold.validation_end,
                "Purged_Correctly": bool(
                    fold.train_exit_end
                    < fold.validation_start
                ),
            }
        )

    return pd.DataFrame(records)


def _fit_model(
    model_name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    fitted = clone(model)

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


def probability_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Classification and probability-quality metrics."""
    y_array = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Positive_Rate_%": float(y_array.mean() * 100),
        "Predicted_Positive_Rate_%": float(
            predictions.mean() * 100
        ),
        "PR_AUC": float(
            average_precision_score(
                y_array,
                probabilities,
            )
        ),
        "Brier": float(
            brier_score_loss(
                y_array,
                probabilities,
            )
        ),
        "Log_Loss": float(
            log_loss(
                y_array,
                probabilities,
                labels=[0, 1],
            )
        ),
        "Precision_0_50": float(
            precision_score(
                y_array,
                predictions,
                zero_division=0,
            )
        ),
        "Recall_0_50": float(
            recall_score(
                y_array,
                predictions,
                zero_division=0,
            )
        ),
        "F1_0_50": float(
            f1_score(
                y_array,
                predictions,
                zero_division=0,
            )
        ),
    }

    metrics["ROC_AUC"] = (
        float(
            roc_auc_score(
                y_array,
                probabilities,
            )
        )
        if len(np.unique(y_array)) == 2
        else np.nan
    )

    return metrics


def cross_validate_models(
    development_data: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    models: Mapping[str, Pipeline],
    folds: Iterable[PurgedFold],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all candidate models using purged expanding folds."""
    features = list(feature_columns)
    fold_records = []
    prediction_frames = []

    for model_name, model in models.items():
        for fold in folds:
            train = development_data.loc[
                fold.train_indices
            ]
            validation = development_data.loc[
                fold.validation_indices
            ]

            fitted = _fit_model(
                model_name=model_name,
                model=model,
                X=train[features],
                y=train[target_column],
            )

            probabilities = fitted.predict_proba(
                validation[features]
            )[:, 1]

            metrics = probability_metrics(
                validation[target_column],
                probabilities,
            )

            metrics.update(
                {
                    "Model": model_name,
                    "Fold": fold.fold,
                    "Train_Count": len(train),
                    "Validation_Count": len(validation),
                    "Validation_Start": fold.validation_start,
                    "Validation_End": fold.validation_end,
                }
            )
            fold_records.append(metrics)

            predictions = validation[
                [
                    "Ticker",
                    "Signal_Date",
                    "Exit_Date",
                    target_column,
                    "Net_Return_%",
                    "R_Multiple",
                ]
            ].copy()
            predictions["Model"] = model_name
            predictions["Fold"] = fold.fold
            predictions["Probability"] = probabilities
            prediction_frames.append(predictions)

    fold_metrics = pd.DataFrame(fold_records)
    oof_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    return fold_metrics, oof_predictions


def summarize_cv_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fold metrics while preserving temporal variability."""
    metric_columns = [
        "ROC_AUC",
        "PR_AUC",
        "Brier",
        "Log_Loss",
        "Precision_0_50",
        "Recall_0_50",
        "F1_0_50",
    ]

    summary = (
        fold_metrics.groupby("Model")[metric_columns]
        .agg(["mean", "std", "min"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    return summary.reset_index().sort_values(
        [
            "PR_AUC_mean",
            "Brier_mean",
        ],
        ascending=[False, True],
    )


def train_on_development_predict_validation(
    development_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    models: Mapping[str, Pipeline],
    cutoff_date: str | pd.Timestamp = "2023-01-01",
    embargo_days: int = 5,
) -> tuple[
    dict[str, Pipeline],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Train purged development models and score the validation period."""
    features = list(feature_columns)
    cutoff = pd.Timestamp(cutoff_date)
    purge_boundary = cutoff - pd.Timedelta(
        days=embargo_days
    )

    eligible_development = development_data.loc[
        pd.to_datetime(
            development_data["Exit_Date"]
        ) < purge_boundary
    ].copy()

    if eligible_development.empty:
        raise ValueError(
            "Validation öncesi uygun development örneği yok."
        )

    fitted_models: dict[str, Pipeline] = {}
    metric_records = []
    prediction_frames = []

    for model_name, model in models.items():
        fitted = _fit_model(
            model_name=model_name,
            model=model,
            X=eligible_development[features],
            y=eligible_development[target_column],
        )
        fitted_models[model_name] = fitted

        probabilities = fitted.predict_proba(
            validation_data[features]
        )[:, 1]

        metrics = probability_metrics(
            validation_data[target_column],
            probabilities,
        )
        metrics.update(
            {
                "Model": model_name,
                "Train_Count": len(
                    eligible_development
                ),
                "Validation_Count": len(
                    validation_data
                ),
                "Train_Positive_Rate_%": (
                    eligible_development[
                        target_column
                    ].mean()
                    * 100
                ),
            }
        )
        metric_records.append(metrics)

        predictions = validation_data.copy()
        predictions["Model"] = model_name
        predictions["Probability"] = probabilities
        prediction_frames.append(predictions)

    validation_metrics = pd.DataFrame(
        metric_records
    ).sort_values(
        ["PR_AUC", "Brier"],
        ascending=[False, True],
    )

    validation_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    return (
        fitted_models,
        validation_metrics,
        validation_predictions,
    )


def event_threshold_table(
    predictions: pd.DataFrame,
    probability_column: str = "Probability",
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate event quality above a range of probability thresholds."""
    if thresholds is None:
        thresholds = np.arange(
            0.25,
            0.751,
            0.025,
        )

    required = {
        probability_column,
        "Meta_Label",
        "Net_Return_%",
        "R_Multiple",
    }
    missing = required.difference(predictions.columns)

    if missing:
        raise KeyError(
            f"Eşik analizi için eksik sütunlar: {sorted(missing)}"
        )

    baseline_positive_rate = (
        predictions["Meta_Label"].mean()
    )
    baseline_average_return = (
        predictions["Net_Return_%"].mean()
    )
    records = []

    for threshold in thresholds:
        selected = predictions.loc[
            predictions[probability_column] >= threshold
        ].copy()

        if selected.empty:
            continue

        positive_returns = selected.loc[
            selected["Net_Return_%"] > 0,
            "Net_Return_%",
        ].sum()

        negative_returns = selected.loc[
            selected["Net_Return_%"] < 0,
            "Net_Return_%",
        ].sum()

        profit_factor = (
            positive_returns / abs(negative_returns)
            if negative_returns < 0
            else np.inf
        )

        records.append(
            {
                "Threshold": float(threshold),
                "Selected_Count": len(selected),
                "Selection_Rate_%": (
                    len(selected) / len(predictions) * 100
                ),
                "Precision_%": (
                    selected["Meta_Label"].mean() * 100
                ),
                "Precision_Lift_pp": (
                    selected["Meta_Label"].mean()
                    - baseline_positive_rate
                )
                * 100,
                "Average_Return_%": (
                    selected["Net_Return_%"].mean()
                ),
                "Average_Return_Lift_pp": (
                    selected["Net_Return_%"].mean()
                    - baseline_average_return
                ),
                "Median_Return_%": (
                    selected["Net_Return_%"].median()
                ),
                "Average_R_Multiple": (
                    selected["R_Multiple"].mean()
                ),
                "Median_R_Multiple": (
                    selected["R_Multiple"].median()
                ),
                "Profit_Factor": profit_factor,
                "Average_Holding_Bars": (
                    selected["Holding_Bars"].mean()
                )
                if "Holding_Bars" in selected.columns
                else np.nan,
            }
        )

    return pd.DataFrame(records)


def validation_permutation_importance(
    fitted_model: Pipeline,
    validation_data: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "Meta_Label",
    scoring: str = "average_precision",
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate validation-set permutation importance."""
    features = list(feature_columns)

    result = permutation_importance(
        fitted_model,
        validation_data[features],
        validation_data[target_column],
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "Feature": features,
                "Importance_Mean": (
                    result.importances_mean
                ),
                "Importance_Std": (
                    result.importances_std
                ),
            }
        )
        .sort_values(
            "Importance_Mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_candidate_model(
    model: Pipeline,
    model_name: str,
    feature_columns: Iterable[str],
    output_directory: str | Path,
    training_end: str | pd.Timestamp,
    target_column: str = "Meta_Label",
) -> tuple[Path, Path]:
    """Persist a candidate model and its audit metadata."""
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        output_dir
        / "meta_label_candidate_model.joblib"
    )
    metadata_path = (
        output_dir
        / "meta_label_candidate_metadata.json"
    )

    dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "training_end": pd.Timestamp(
            training_end
        ).isoformat(),
        "threshold_status": (
            "PORTFOLIO_BACKTEST_REQUIRED"
        ),
        "notes": (
            "Probability threshold is not final. "
            "Validate candidate thresholds inside the "
            "event-driven portfolio backtest."
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return model_path, metadata_path
