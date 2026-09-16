"""RS126 Enhanced + locked ML hybrid research utilities.

This module does not modify the production Baseline, RS126 Enhanced or
ML Challenger rules. It performs one controlled historical comparison:

1. Baseline Robot
2. Existing locked ML Challenger on Baseline candidates
3. RS126 Enhanced
4. Existing locked ML filter on RS126 Enhanced candidates
5. BIST100 Gross benchmark

The ML target, model family and probability threshold are read from the
already-locked Validation decision. No threshold or model type is retuned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.benchmark import (
    active_performance_metrics,
    align_equity_curves,
    build_benchmark_equity,
    calendar_return_table,
)
from src.config import (
    PortfolioConfig,
    StrategyConfig,
)
from src.enhanced_baseline import (
    RULE_DESCRIPTION,
    RULE_NAME,
    build_rs126_enhanced_prices,
)
from src.metrics import portfolio_metrics
from src.ml_portfolio import (
    MLFilterConfig,
    apply_ml_filter,
    evaluate_filter,
)
from src.ml_walkforward import (
    WalkForwardConfig,
    generate_walk_forward_probabilities,
)


PORTFOLIO_ORDER = [
    "BIST100_Gross",
    "Baseline_Robot",
    "RS126_Enhanced",
    "ML_Challenger",
    "RS126_Enhanced_ML",
]


@dataclass(frozen=True)
class LockedMLSpec:
    """Previously accepted and locked ML deployment decision."""

    target: str
    model_name: str
    filter_name: str
    probability_threshold: float | None
    keep_top_fraction: float | None
    decision_path: Path
    acceptance_path: Path


@dataclass
class HybridExperimentArtifacts:
    """In-memory outputs of the four-strategy hybrid comparison."""

    metrics: pd.DataFrame
    equity: pd.DataFrame
    yearly: pd.DataFrame
    active: pd.DataFrame
    pairwise: pd.DataFrame
    funnel: pd.DataFrame
    acceptance: pd.DataFrame
    training_log: pd.DataFrame
    baseline_probability_prices: pd.DataFrame
    enhanced_probability_prices: pd.DataFrame
    outputs: dict[str, dict[str, pd.DataFrame]]
    metadata: dict[str, Any]


def load_locked_ml_spec(
    project_root: str | Path,
) -> LockedMLSpec:
    """Read the accepted target/model/filter without loading a static model."""
    root = Path(project_root)

    decision_path = (
        root
        / "models"
        / "alternative_target_ml_decision.json"
    )
    acceptance_path = (
        root
        / "results"
        / "ml"
        / "alternative_targets_validation_acceptance.csv"
    )

    missing = [
        path
        for path in (
            decision_path,
            acceptance_path,
        )
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Kilitli ML kararı için eksik dosyalar: "
            + ", ".join(str(path) for path in missing)
        )

    decision = json.loads(
        decision_path.read_text(
            encoding="utf-8"
        )
    )

    if not bool(
        decision.get(
            "ml_accepted_on_validation",
            False,
        )
    ):
        raise RuntimeError(
            "Kilitli Validation kararında ML kabul edilmemiş."
        )

    target = decision.get("selected_target")
    model_name = decision.get("selected_model")
    filter_name = decision.get("selected_filter")

    if not all(
        [
            target,
            model_name,
            filter_name,
        ]
    ):
        raise ValueError(
            "ML karar dosyasında target/model/filter eksik."
        )

    acceptance = pd.read_csv(
        acceptance_path
    )

    required = {
        "Target",
        "Model",
        "Filter_Name",
        "ML_Accepted",
        "Probability_Threshold",
        "Keep_Top_Fraction",
    }
    missing_columns = required.difference(
        acceptance.columns
    )

    if missing_columns:
        raise KeyError(
            "ML acceptance tablosunda eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )

    accepted = (
        acceptance["ML_Accepted"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    rows = acceptance.loc[
        acceptance["Target"].eq(target)
        & acceptance["Model"].eq(model_name)
        & acceptance["Filter_Name"].eq(
            filter_name
        )
        & accepted
    ]

    if len(rows) != 1:
        raise ValueError(
            "Kilitli ML kararı için acceptance tablosunda "
            "tek satır bulunmalıdır."
        )

    row = rows.iloc[0]

    probability_threshold = (
        None
        if pd.isna(
            row["Probability_Threshold"]
        )
        else float(
            row["Probability_Threshold"]
        )
    )
    keep_top_fraction = (
        None
        if pd.isna(
            row["Keep_Top_Fraction"]
        )
        else float(
            row["Keep_Top_Fraction"]
        )
    )

    return LockedMLSpec(
        target=str(target),
        model_name=str(model_name),
        filter_name=str(filter_name),
        probability_threshold=(
            probability_threshold
        ),
        keep_top_fraction=keep_top_fraction,
        decision_path=decision_path,
        acceptance_path=acceptance_path,
    )


def locked_filter_config(
    spec: LockedMLSpec,
    name: str,
) -> MLFilterConfig:
    """Create a filter with the already-locked Validation settings."""
    return MLFilterConfig(
        name=name,
        probability_threshold=(
            spec.probability_threshold
        ),
        keep_top_fraction=(
            spec.keep_top_fraction
        ),
    )


def build_hybrid_probability_universes(
    featured_baseline_prices: pd.DataFrame,
    events: pd.DataFrame,
    market_features: pd.DataFrame,
    spec: LockedMLSpec,
    walk_forward_config: WalkForwardConfig,
    strategy_config: StrategyConfig,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Fit the locked monthly model once and reuse scores for Enhanced rows.

    RS126 Enhanced only demotes Baseline AL rows; it never creates a new AL.
    Therefore every Enhanced AL candidate already has a Baseline walk-forward
    probability. Reusing that probability avoids fitting the same model twice
    and preserves the original model feature semantics.
    """
    baseline_input = (
        featured_baseline_prices.copy()
    )
    baseline_input["Date"] = pd.to_datetime(
        baseline_input["Date"]
    )
    baseline_input[
        "Baseline_Signal"
    ] = baseline_input["Signal"]
    baseline_input[
        "Baseline_Score"
    ] = pd.to_numeric(
        baseline_input["Score"],
        errors="coerce",
    )

    (
        baseline_probability_prices,
        training_log,
    ) = generate_walk_forward_probabilities(
        featured_prices=baseline_input,
        events=events,
        model_name=spec.model_name,
        target_column=spec.target,
        feature_columns=feature_columns,
        config=walk_forward_config,
    )

    enhanced_probability_prices = (
        build_rs126_enhanced_prices(
            baseline_prices=(
                baseline_probability_prices
            ),
            market_features=market_features,
            strategy_config=strategy_config,
        )
    )

    period_mask = pd.to_datetime(
        baseline_probability_prices["Date"]
    ).between(
        pd.Timestamp(
            walk_forward_config.start
        ),
        (
            pd.Timestamp(
                walk_forward_config.end
            )
            if walk_forward_config.end
            else pd.to_datetime(
                baseline_probability_prices[
                    "Date"
                ]
            ).max()
        ),
        inclusive="both",
    )

    baseline_al = (
        period_mask
        & baseline_probability_prices[
            "Signal"
        ].eq("AL")
    )
    enhanced_al = (
        period_mask
        & enhanced_probability_prices[
            "Signal"
        ].eq("AL")
    )

    promoted = (
        enhanced_al
        & ~baseline_al
    )

    if promoted.any():
        examples = (
            enhanced_probability_prices.loc[
                promoted,
                ["Date", "Ticker"],
            ]
            .head(20)
            .astype(str)
            .to_dict(orient="records")
        )

        raise RuntimeError(
            "RS126 Enhanced beklenmedik biçimde yeni AL "
            f"üretti. Örnekler: {examples}"
        )

    missing_probability = (
        enhanced_al
        & enhanced_probability_prices[
            "ML_Probability"
        ].isna()
    )

    if missing_probability.any():
        examples = (
            enhanced_probability_prices.loc[
                missing_probability,
                ["Date", "Ticker"],
            ]
            .head(20)
            .astype(str)
            .to_dict(orient="records")
        )

        raise RuntimeError(
            "Enhanced AL adaylarında ML olasılığı eksik. "
            f"Örnekler: {examples}"
        )

    return (
        baseline_probability_prices,
        enhanced_probability_prices,
        training_log,
    )


def _metric_record(
    portfolio_name: str,
    universe: str,
    ml_applied: bool,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    columns = [
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
        "Outlier_Count",
        "Exposure_%",
        "Average_Open_Positions",
        "Max_Open_Positions",
        "Average_Invested_%",
        "Signal_Pass_Rate_%",
    ]

    return {
        "Portfolio": portfolio_name,
        "Candidate_Universe": universe,
        "ML_Applied": ml_applied,
        **{
            column: metrics.get(column)
            for column in columns
            if column in metrics
        },
    }


def _equity_view(
    equity: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    return (
        equity[
            [
                "Date",
                "Equity",
            ]
        ]
        .rename(
            columns={
                "Equity": name,
            }
        )
        .assign(
            Date=lambda frame: pd.to_datetime(
                frame["Date"]
            )
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _candidate_funnel(
    baseline_prices: pd.DataFrame,
    enhanced_prices: pd.DataFrame,
    baseline_ml_filtered: pd.DataFrame,
    hybrid_filtered: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Summarize how many signal rows survive each filter stage."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    period = pd.to_datetime(
        baseline_prices["Date"]
    ).between(
        start_ts,
        end_ts,
        inclusive="both",
    )

    baseline_al = (
        period
        & baseline_prices["Signal"].eq("AL")
    )
    enhanced_al = (
        period
        & enhanced_prices["Signal"].eq("AL")
    )
    ml_pass = (
        period
        & baseline_ml_filtered[
            "ML_Filter_Passed"
        ]
    )
    hybrid_pass = (
        period
        & hybrid_filtered[
            "ML_Filter_Passed"
        ]
    )

    baseline_count = int(
        baseline_al.sum()
    )
    enhanced_count = int(
        enhanced_al.sum()
    )
    ml_count = int(
        ml_pass.sum()
    )
    hybrid_count = int(
        hybrid_pass.sum()
    )

    rows = [
        {
            "Stage": "Baseline AL",
            "Signal_Rows": baseline_count,
            "Pass_Rate_vs_Baseline_%": 100.0,
            "Removed_From_Previous": 0,
        },
        {
            "Stage": "RS126 Enhanced AL",
            "Signal_Rows": enhanced_count,
            "Pass_Rate_vs_Baseline_%": (
                enhanced_count
                / baseline_count
                * 100
                if baseline_count
                else np.nan
            ),
            "Removed_From_Previous": (
                baseline_count
                - enhanced_count
            ),
        },
        {
            "Stage": "Baseline + ML",
            "Signal_Rows": ml_count,
            "Pass_Rate_vs_Baseline_%": (
                ml_count
                / baseline_count
                * 100
                if baseline_count
                else np.nan
            ),
            "Removed_From_Previous": (
                baseline_count
                - ml_count
            ),
        },
        {
            "Stage": "RS126 Enhanced + ML",
            "Signal_Rows": hybrid_count,
            "Pass_Rate_vs_Baseline_%": (
                hybrid_count
                / baseline_count
                * 100
                if baseline_count
                else np.nan
            ),
            "Removed_From_Previous": (
                enhanced_count
                - hybrid_count
            ),
        },
    ]

    return pd.DataFrame(rows)


def pairwise_hybrid_comparison(
    metrics: pd.DataFrame,
    focus_name: str = (
        "RS126_Enhanced_ML"
    ),
) -> pd.DataFrame:
    """Compare the Hybrid against every meaningful reference."""
    if "Portfolio" not in metrics.columns:
        raise KeyError(
            "Metrics tablosunda Portfolio sütunu yok."
        )

    indexed = metrics.set_index("Portfolio")

    if focus_name not in indexed.index:
        raise KeyError(
            f"Focus portföy bulunamadı: {focus_name}"
        )

    focus = indexed.loc[focus_name]
    references = [
        "BIST100_Gross",
        "Baseline_Robot",
        "RS126_Enhanced",
        "ML_Challenger",
    ]

    records = []

    for reference_name in references:
        if reference_name not in indexed.index:
            continue

        reference = indexed.loc[
            reference_name
        ]

        focus_trades = pd.to_numeric(
            pd.Series(
                [focus.get("Trade_Count")]
            ),
            errors="coerce",
        ).iloc[0]
        reference_trades = pd.to_numeric(
            pd.Series(
                [reference.get("Trade_Count")]
            ),
            errors="coerce",
        ).iloc[0]

        records.append(
            {
                "Focus": focus_name,
                "Reference": reference_name,
                "End_Value_Difference_TL": (
                    focus.get("End_Value", np.nan)
                    - reference.get(
                        "End_Value",
                        np.nan,
                    )
                ),
                "CAGR_Difference_pp": (
                    focus.get("CAGR_%", np.nan)
                    - reference.get(
                        "CAGR_%",
                        np.nan,
                    )
                ),
                "Max_DD_Improvement_pp": (
                    focus.get(
                        "Max_Drawdown_%",
                        np.nan,
                    )
                    - reference.get(
                        "Max_Drawdown_%",
                        np.nan,
                    )
                ),
                "Profit_Factor_Difference": (
                    focus.get(
                        "Profit_Factor",
                        np.nan,
                    )
                    - reference.get(
                        "Profit_Factor",
                        np.nan,
                    )
                ),
                "Sharpe_Difference": (
                    focus.get("Sharpe", np.nan)
                    - reference.get(
                        "Sharpe",
                        np.nan,
                    )
                ),
                "Calmar_Difference": (
                    focus.get("Calmar", np.nan)
                    - reference.get(
                        "Calmar",
                        np.nan,
                    )
                ),
                "Trade_Fraction": (
                    focus_trades
                    / reference_trades
                    if (
                        pd.notna(focus_trades)
                        and pd.notna(
                            reference_trades
                        )
                        and reference_trades > 0
                    )
                    else np.nan
                ),
                "Exposure_Difference_pp": (
                    focus.get(
                        "Exposure_%",
                        np.nan,
                    )
                    - reference.get(
                        "Exposure_%",
                        np.nan,
                    )
                ),
            }
        )

    return pd.DataFrame(records)


def hybrid_historical_screen(
    metrics: pd.DataFrame,
    focus_name: str = (
        "RS126_Enhanced_ML"
    ),
    primary_reference: str = (
        "RS126_Enhanced"
    ),
    minimum_trade_fraction: float = 0.70,
    minimum_return_case_cagr_pp: float = 1.50,
    minimum_return_case_calmar_pct: float = 5.0,
    minimum_risk_case_cagr_pp: float = -1.0,
    minimum_risk_case_dd_improvement_pp: float = 2.0,
    minimum_risk_case_calmar_pct: float = 8.0,
    minimum_profit_factor_ratio: float = 0.95,
) -> pd.DataFrame:
    """Apply predeclared criteria relative to RS126 Enhanced.

    This is a historical screen, not a new independent Audit, because the
    RS126 rule was already selected after examining the 2025+ period.
    """
    indexed = metrics.set_index("Portfolio")

    for name in (
        focus_name,
        primary_reference,
    ):
        if name not in indexed.index:
            raise KeyError(
                f"Portföy bulunamadı: {name}"
            )

    focus = indexed.loc[focus_name]
    reference = indexed.loc[
        primary_reference
    ]

    cagr_delta = (
        focus["CAGR_%"]
        - reference["CAGR_%"]
    )
    dd_improvement = (
        focus["Max_Drawdown_%"]
        - reference["Max_Drawdown_%"]
    )
    pf_ratio = (
        focus["Profit_Factor"]
        / reference["Profit_Factor"]
        if reference["Profit_Factor"]
        else np.nan
    )
    calmar_improvement_pct = (
        (
            focus["Calmar"]
            / abs(reference["Calmar"])
            - 1
        )
        * 100
        if (
            pd.notna(reference["Calmar"])
            and abs(reference["Calmar"])
            > 1e-12
        )
        else np.nan
    )
    trade_fraction = (
        focus["Trade_Count"]
        / reference["Trade_Count"]
        if reference["Trade_Count"]
        else np.nan
    )

    return_case = bool(
        cagr_delta
        >= minimum_return_case_cagr_pp
        and dd_improvement >= 0.0
        and calmar_improvement_pct
        >= minimum_return_case_calmar_pct
        and focus["Profit_Factor"]
        >= reference["Profit_Factor"]
        and trade_fraction
        >= minimum_trade_fraction
    )

    risk_case = bool(
        cagr_delta
        >= minimum_risk_case_cagr_pp
        and dd_improvement
        >= minimum_risk_case_dd_improvement_pp
        and calmar_improvement_pct
        >= minimum_risk_case_calmar_pct
        and pf_ratio
        >= minimum_profit_factor_ratio
        and trade_fraction
        >= minimum_trade_fraction
    )

    return pd.DataFrame(
        [
            {
                "Focus": focus_name,
                "Primary_Reference": (
                    primary_reference
                ),
                "CAGR_Difference_pp": (
                    cagr_delta
                ),
                "Max_DD_Improvement_pp": (
                    dd_improvement
                ),
                "Profit_Factor_Ratio": (
                    pf_ratio
                ),
                "Calmar_Improvement_%": (
                    calmar_improvement_pct
                ),
                "Trade_Fraction": (
                    trade_fraction
                ),
                "Return_Case_Passed": (
                    return_case
                ),
                "Risk_Case_Passed": (
                    risk_case
                ),
                "Historical_Screen_Passed": (
                    return_case
                    or risk_case
                ),
                "Independent_Audit": False,
                "Decision": (
                    "Forward paper-trading adayı"
                    if (
                        return_case
                        or risk_case
                    )
                    else "RS126 Enhanced korunur"
                ),
            }
        ]
    )


def evaluate_hybrid_experiment(
    baseline_probability_prices: pd.DataFrame,
    enhanced_probability_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    spec: LockedMLSpec,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    training_log: pd.DataFrame,
) -> HybridExperimentArtifacts:
    """Run the four strategies and BIST100 over exactly the same dates."""
    baseline_filter = MLFilterConfig(
        name="Baseline_Robot"
    )
    ml_filter = locked_filter_config(
        spec,
        "ML_Challenger",
    )
    enhanced_filter = MLFilterConfig(
        name="RS126_Enhanced"
    )
    hybrid_filter = locked_filter_config(
        spec,
        "RS126_Enhanced_ML",
    )

    (
        baseline_metrics,
        baseline_equity,
        baseline_trades,
    ) = evaluate_filter(
        probability_prices=(
            baseline_probability_prices
        ),
        filter_config=baseline_filter,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
        start=start,
        end=end,
    )

    (
        ml_metrics,
        ml_equity,
        ml_trades,
    ) = evaluate_filter(
        probability_prices=(
            baseline_probability_prices
        ),
        filter_config=ml_filter,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
        start=start,
        end=end,
    )

    (
        enhanced_metrics,
        enhanced_equity,
        enhanced_trades,
    ) = evaluate_filter(
        probability_prices=(
            enhanced_probability_prices
        ),
        filter_config=enhanced_filter,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
        start=start,
        end=end,
    )

    (
        hybrid_metrics,
        hybrid_equity,
        hybrid_trades,
    ) = evaluate_filter(
        probability_prices=(
            enhanced_probability_prices
        ),
        filter_config=hybrid_filter,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
        start=start,
        end=end,
    )

    outputs = {
        "Baseline_Robot": {
            "equity": baseline_equity,
            "trades": baseline_trades,
        },
        "ML_Challenger": {
            "equity": ml_equity,
            "trades": ml_trades,
        },
        "RS126_Enhanced": {
            "equity": enhanced_equity,
            "trades": enhanced_trades,
        },
        "RS126_Enhanced_ML": {
            "equity": hybrid_equity,
            "trades": hybrid_trades,
        },
    }

    benchmark_equity = (
        build_benchmark_equity(
            market_prices=market_prices,
            comparison_dates=(
                baseline_equity["Date"]
            ),
            initial_capital=(
                portfolio_config.initial_capital
            ),
            include_costs=False,
            benchmark_name="BIST100",
        )
    )

    benchmark_metrics = portfolio_metrics(
        benchmark_equity,
        pd.DataFrame(
            columns=["Return"]
        ),
    )

    metrics = pd.DataFrame(
        [
            _metric_record(
                "Baseline_Robot",
                "Baseline AL",
                False,
                baseline_metrics,
            ),
            _metric_record(
                "RS126_Enhanced",
                "RS126 Enhanced AL",
                False,
                enhanced_metrics,
            ),
            _metric_record(
                "ML_Challenger",
                "Baseline AL",
                True,
                ml_metrics,
            ),
            _metric_record(
                "RS126_Enhanced_ML",
                "RS126 Enhanced AL",
                True,
                hybrid_metrics,
            ),
            _metric_record(
                "BIST100_Gross",
                "Passive benchmark",
                False,
                benchmark_metrics,
            ),
        ]
    )

    order_map = {
        name: index
        for index, name in enumerate(
            PORTFOLIO_ORDER
        )
    }
    metrics["Display_Order"] = (
        metrics["Portfolio"]
        .map(order_map)
        .fillna(99)
    )
    metrics = (
        metrics.sort_values(
            "Display_Order"
        )
        .drop(
            columns=["Display_Order"]
        )
        .reset_index(drop=True)
    )

    equity = _equity_view(
        baseline_equity,
        "Baseline_Robot",
    )

    for name, frame in [
        (
            "RS126_Enhanced",
            enhanced_equity,
        ),
        (
            "ML_Challenger",
            ml_equity,
        ),
        (
            "RS126_Enhanced_ML",
            hybrid_equity,
        ),
        (
            "BIST100_Gross",
            benchmark_equity,
        ),
    ]:
        equity = equity.merge(
            _equity_view(
                frame,
                name,
            ),
            on="Date",
            how="inner",
            validate="one_to_one",
        )

    yearly = calendar_return_table(
        {
            name: frame["equity"]
            for name, frame in outputs.items()
        }
        | {
            "BIST100_Gross": (
                benchmark_equity
            )
        },
        frequency="YE",
    )
    yearly["Year"] = (
        yearly["Date"].dt.year
    )
    yearly = (
        yearly.pivot_table(
            index="Year",
            columns="Portfolio",
            values="Return_%",
            aggfunc="first",
        )
        .reset_index()
    )

    active_records = []

    for portfolio_name, frames in (
        outputs.items()
    ):
        aligned = align_equity_curves(
            frames["equity"],
            benchmark_equity,
        )
        active = active_performance_metrics(
            aligned
        )
        active["Portfolio"] = (
            portfolio_name
        )
        active_records.append(active)

    active = pd.DataFrame(
        active_records
    )

    baseline_ml_filtered = apply_ml_filter(
        baseline_probability_prices,
        ml_filter,
    )
    hybrid_filtered = apply_ml_filter(
        enhanced_probability_prices,
        hybrid_filter,
    )

    funnel = _candidate_funnel(
        baseline_prices=(
            baseline_probability_prices
        ),
        enhanced_prices=(
            enhanced_probability_prices
        ),
        baseline_ml_filtered=(
            baseline_ml_filtered
        ),
        hybrid_filtered=hybrid_filtered,
        start=start,
        end=end,
    )

    pairwise = pairwise_hybrid_comparison(
        metrics
    )
    acceptance = hybrid_historical_screen(
        metrics
    )

    metadata = {
        "experiment": (
            "RS126 Enhanced candidate universe "
            "+ locked monthly walk-forward ML"
        ),
        "start": str(
            pd.Timestamp(start).date()
        ),
        "end": str(
            pd.Timestamp(end).date()
        ),
        "locked_ml_spec": {
            **asdict(spec),
            "decision_path": str(
                spec.decision_path
            ),
            "acceptance_path": str(
                spec.acceptance_path
            ),
        },
        "rs126_rule_name": RULE_NAME,
        "rs126_rule": RULE_DESCRIPTION,
        "probability_reuse": (
            "Enhanced AL is a subset of Baseline AL; "
            "the same leakage-safe monthly probability "
            "is reused by Date/Ticker."
        ),
        "threshold_retuned": False,
        "model_type_retuned": False,
        "independent_audit": False,
        "methodological_warning": (
            "RS126 Enhanced was selected after examining "
            "the 2025+ historical period. Therefore this "
            "hybrid result is a historical screen and must "
            "be confirmed with forward paper trading."
        ),
    }

    return HybridExperimentArtifacts(
        metrics=metrics,
        equity=equity,
        yearly=yearly,
        active=active,
        pairwise=pairwise,
        funnel=funnel,
        acceptance=acceptance,
        training_log=training_log,
        baseline_probability_prices=(
            baseline_probability_prices
        ),
        enhanced_probability_prices=(
            enhanced_probability_prices
        ),
        outputs=outputs,
        metadata=metadata,
    )


def save_hybrid_artifacts(
    output_directory: str | Path,
    artifacts: HybridExperimentArtifacts,
) -> dict[str, Path]:
    """Save local research outputs under the ignored results directory."""
    output = Path(output_directory)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "metrics": (
            output
            / "hybrid_comparison_metrics.csv"
        ),
        "equity": (
            output
            / "hybrid_equity.parquet"
        ),
        "yearly": (
            output
            / "hybrid_yearly.csv"
        ),
        "active": (
            output
            / "hybrid_active_metrics.csv"
        ),
        "pairwise": (
            output
            / "hybrid_pairwise_comparison.csv"
        ),
        "funnel": (
            output
            / "hybrid_candidate_funnel.csv"
        ),
        "acceptance": (
            output
            / "hybrid_historical_screen.csv"
        ),
        "training_log": (
            output
            / "hybrid_walk_forward_training_log.csv"
        ),
        "probabilities": (
            output
            / "hybrid_probabilities.parquet"
        ),
        "metadata": (
            output
            / "hybrid_metadata.json"
        ),
    }

    artifacts.metrics.to_csv(
        paths["metrics"],
        index=False,
    )
    artifacts.equity.to_parquet(
        paths["equity"],
        index=False,
    )
    artifacts.yearly.to_csv(
        paths["yearly"],
        index=False,
    )
    artifacts.active.to_csv(
        paths["active"],
        index=False,
    )
    artifacts.pairwise.to_csv(
        paths["pairwise"],
        index=False,
    )
    artifacts.funnel.to_csv(
        paths["funnel"],
        index=False,
    )
    artifacts.acceptance.to_csv(
        paths["acceptance"],
        index=False,
    )
    artifacts.training_log.to_csv(
        paths["training_log"],
        index=False,
    )

    probability_columns = [
        column
        for column in (
            "Date",
            "Ticker",
            "Baseline_Score",
            "Enhanced_Score",
            "RS_126",
            "RS126_Penalty",
            "Baseline_Signal",
            "Signal",
            "ML_Probability",
            "WF_Training_End",
            "WF_Training_Event_Count",
            "WF_Block_Start",
            "WF_Block_End",
        )
        if column
        in artifacts.enhanced_probability_prices.columns
    ]

    artifacts.enhanced_probability_prices[
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
            artifacts.metadata,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return paths
