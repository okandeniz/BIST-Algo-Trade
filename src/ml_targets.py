"""Alternative meta-label target experiments for the Robot strategy."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.ml_portfolio import MLFilterConfig


def add_alternative_targets(
    dataset: pd.DataFrame,
    one_r_threshold: float = 1.0,
    big_winner_threshold: float = 2.0,
) -> pd.DataFrame:
    """Create 1R and big-winner labels from realized R multiples."""
    required = {"R_Multiple"}
    missing = required.difference(dataset.columns)

    if missing:
        raise KeyError(
            f"Alternatif hedefler için eksik sütunlar: {sorted(missing)}"
        )

    result = dataset.copy()

    result["Meta_Label_1R"] = (
        result["R_Multiple"] >= one_r_threshold
    ).astype("int8")

    result["Big_Winner_Label_2R"] = (
        result["R_Multiple"] >= big_winner_threshold
    ).astype("int8")

    return result


def target_diagnostics(
    dataset: pd.DataFrame,
    target_columns: Iterable[str],
) -> pd.DataFrame:
    """Summarize target prevalence and realized return by period."""
    records = []

    for target_column in target_columns:
        if target_column not in dataset.columns:
            raise KeyError(
                f"Target bulunamadı: {target_column}"
            )

        grouped = (
            dataset.groupby("Period")
            .agg(
                Event_Count=(target_column, "size"),
                Positive_Count=(target_column, "sum"),
                Positive_Rate=(target_column, "mean"),
                Average_Return_=("Net_Return_%", "mean"),
                Median_Return_=("Net_Return_%", "median"),
                Average_R_Multiple=("R_Multiple", "mean"),
            )
            .reset_index()
        )

        grouped["Target"] = target_column
        grouped["Positive_Rate_%"] = (
            grouped["Positive_Rate"] * 100
        )
        records.append(grouped)

    return pd.concat(
        records,
        ignore_index=True,
    )


def summarize_cv_lift(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate purged-CV metrics relative to each fold's base rate."""
    required = {
        "Model",
        "Fold",
        "PR_AUC",
        "ROC_AUC",
        "Brier",
        "Log_Loss",
        "Positive_Rate_%",
    }
    missing = required.difference(fold_metrics.columns)

    if missing:
        raise KeyError(
            f"CV lift özeti için eksik sütunlar: {sorted(missing)}"
        )

    work = fold_metrics.copy()
    work["PR_AUC_Base_Rate"] = (
        work["Positive_Rate_%"] / 100
    )
    work["PR_AUC_Lift"] = (
        work["PR_AUC"] - work["PR_AUC_Base_Rate"]
    )

    summary = (
        work.groupby("Model")
        .agg(
            PR_AUC_Mean=("PR_AUC", "mean"),
            PR_AUC_Min=("PR_AUC", "min"),
            PR_AUC_Lift_Mean=("PR_AUC_Lift", "mean"),
            PR_AUC_Lift_Min=("PR_AUC_Lift", "min"),
            ROC_AUC_Mean=("ROC_AUC", "mean"),
            ROC_AUC_Min=("ROC_AUC", "min"),
            Brier_Mean=("Brier", "mean"),
            Brier_Max=("Brier", "max"),
            Log_Loss_Mean=("Log_Loss", "mean"),
            Fold_Count=("Fold", "nunique"),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "PR_AUC_Lift_Mean",
            "PR_AUC_Lift_Min",
            "Brier_Mean",
            "ROC_AUC_Mean",
        ],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def select_cv_champion(
    cv_lift_summary: pd.DataFrame,
) -> str:
    """Choose the model using Development purged-CV only."""
    if cv_lift_summary.empty:
        raise ValueError("CV özeti boş.")

    return str(cv_lift_summary.iloc[0]["Model"])


def target_threshold_table(
    predictions: pd.DataFrame,
    target_column: str,
    probability_column: str = "Probability",
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate event quality above probability thresholds."""
    if thresholds is None:
        thresholds = np.arange(
            0.10,
            0.901,
            0.025,
        )

    required = {
        target_column,
        probability_column,
        "Net_Return_%",
        "R_Multiple",
        "Holding_Bars",
    }
    missing = required.difference(predictions.columns)

    if missing:
        raise KeyError(
            f"Eşik analizi için eksik sütunlar: {sorted(missing)}"
        )

    baseline_positive_rate = predictions[
        target_column
    ].mean()
    baseline_average_return = predictions[
        "Net_Return_%"
    ].mean()

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
                "Target_Precision_%": (
                    selected[target_column].mean() * 100
                ),
                "Target_Precision_Lift_pp": (
                    selected[target_column].mean()
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
                ),
            }
        )

    return pd.DataFrame(records)


def build_validation_filter_grid(
    validation_probability_prices: pd.DataFrame,
    probability_column: str = "ML_Probability",
    quantiles: Iterable[float] = (
        0.25,
        0.40,
        0.50,
        0.60,
        0.75,
    ),
    daily_top_fractions: Iterable[float] = (
        0.75,
        0.50,
        0.30,
    ),
) -> tuple[list[MLFilterConfig], pd.DataFrame]:
    """Create absolute thresholds from Validation AL-score quantiles."""
    required = {
        "Date",
        "Signal",
        probability_column,
    }
    missing = required.difference(
        validation_probability_prices.columns
    )

    if missing:
        raise KeyError(
            f"Filtre grid'i için eksik sütunlar: {sorted(missing)}"
        )

    probabilities = (
        validation_probability_prices.loc[
            validation_probability_prices["Signal"].eq("AL"),
            probability_column,
        ]
        .dropna()
    )

    if probabilities.empty:
        raise ValueError(
            "Validation AL satırlarında ML olasılığı bulunamadı."
        )

    quantile_rows = []
    filters = [
        MLFilterConfig(
            name="Baseline_Robot",
        )
    ]

    used_thresholds: set[float] = set()

    for quantile in quantiles:
        threshold = float(
            probabilities.quantile(quantile)
        )
        threshold = round(threshold, 6)

        if threshold in used_thresholds:
            continue

        used_thresholds.add(threshold)

        quantile_rows.append(
            {
                "Quantile": float(quantile),
                "Threshold": threshold,
            }
        )

        filters.append(
            MLFilterConfig(
                name=(
                    f"Quantile_Q{int(round(quantile * 100)):02d}"
                ),
                probability_threshold=threshold,
            )
        )

    for fraction in daily_top_fractions:
        filters.append(
            MLFilterConfig(
                name=(
                    f"Daily_Top_{int(round(fraction * 100)):02d}pct"
                ),
                keep_top_fraction=float(fraction),
            )
        )

    return (
        filters,
        pd.DataFrame(quantile_rows),
    )


def combine_acceptance_results(
    acceptance_tables: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Combine target-specific portfolio acceptance results."""
    frames = [
        frame.copy()
        for frame in acceptance_tables
        if not frame.empty
    ]

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def select_global_validation_champion(
    combined_acceptance: pd.DataFrame,
) -> tuple[dict[str, object] | None, bool]:
    """Select one accepted target/model/filter combination."""
    if combined_acceptance.empty:
        return None, False

    accepted = combined_acceptance.loc[
        combined_acceptance["ML_Accepted"]
    ].copy()

    if accepted.empty:
        return None, False

    champion = (
        accepted.sort_values(
            [
                "Calmar",
                "CAGR_%",
                "Profit_Factor",
                "Sharpe",
                "Trade_Count",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
        .to_dict()
    )

    return champion, True
