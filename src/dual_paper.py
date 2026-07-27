"""Dual paper-trading helpers for Baseline Robot and ML Challenger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from joblib import load

from src.config import PortfolioConfig, StrategyConfig
from src.daily_signal import DailyPlan, create_daily_plan
from src.ml_dataset import BASE_FEATURE_COLUMNS
from src.ml_portfolio import (
    MLFilterConfig,
    add_model_probabilities,
    apply_ml_filter,
)
from src.paper_trading import (
    append_closed_trade,
    calculate_fill_position_size,
    mark_to_market,
    record_buy,
    record_sell,
    save_paper_state,
)


@dataclass(frozen=True)
class ChallengerDeployment:
    """Locked ML challenger deployment configuration."""

    target: str
    model_name: str
    filter_name: str
    probability_threshold: float | None
    keep_top_fraction: float | None
    model_path: Path
    decision_path: Path
    acceptance_path: Path


@dataclass
class DualDailyPlan:
    """Daily plans for the baseline and challenger portfolios."""

    signal_date: pd.Timestamp
    baseline: DailyPlan
    challenger: DailyPlan
    buy_comparison: pd.DataFrame
    sell_comparison: pd.DataFrame
    summary_comparison: pd.DataFrame


def load_challenger_deployment(
    project_root: str | Path,
) -> tuple[ChallengerDeployment, Any]:
    """Load the locked challenger decision, threshold and trained model."""
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
    model_path = (
        root
        / "models"
        / "alternative_target_ml_model.joblib"
    )

    missing_paths = [
        path
        for path in (
            decision_path,
            acceptance_path,
            model_path,
        )
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Challenger deployment için eksik dosyalar: "
            + ", ".join(str(path) for path in missing_paths)
        )

    with decision_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        decision = json.load(file)

    if not bool(
        decision.get("ml_accepted_on_validation", False)
    ):
        raise RuntimeError(
            "Validation kararında ML kabul edilmemiş."
        )

    selected_target = decision.get("selected_target")
    selected_model = decision.get("selected_model")
    selected_filter = decision.get("selected_filter")

    if not all(
        [
            selected_target,
            selected_model,
            selected_filter,
        ]
    ):
        raise ValueError(
            "Karar dosyasında target/model/filter alanları eksik."
        )

    acceptance = pd.read_csv(acceptance_path)

    required = {
        "Target",
        "Model",
        "Filter_Name",
        "ML_Accepted",
        "Probability_Threshold",
        "Keep_Top_Fraction",
    }
    missing = required.difference(acceptance.columns)

    if missing:
        raise KeyError(
            "Acceptance dosyasında eksik sütunlar: "
            f"{sorted(missing)}"
        )

    accepted_flag = (
        acceptance["ML_Accepted"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    selected_rows = acceptance.loc[
        acceptance["Target"].eq(selected_target)
        & acceptance["Model"].eq(selected_model)
        & acceptance["Filter_Name"].eq(selected_filter)
        & accepted_flag
    ]

    if len(selected_rows) != 1:
        raise ValueError(
            "Seçilen challenger için acceptance tablosunda "
            "tek bir kabul edilmiş satır bulunmalıdır."
        )

    row = selected_rows.iloc[0]

    probability_threshold = (
        None
        if pd.isna(row["Probability_Threshold"])
        else float(row["Probability_Threshold"])
    )
    keep_top_fraction = (
        None
        if pd.isna(row["Keep_Top_Fraction"])
        else float(row["Keep_Top_Fraction"])
    )

    deployment = ChallengerDeployment(
        target=str(selected_target),
        model_name=str(selected_model),
        filter_name=str(selected_filter),
        probability_threshold=probability_threshold,
        keep_top_fraction=keep_top_fraction,
        model_path=model_path,
        decision_path=decision_path,
        acceptance_path=acceptance_path,
    )

    return deployment, load(model_path)


def deployment_summary(
    deployment: ChallengerDeployment,
) -> pd.DataFrame:
    """Return a readable one-row deployment summary."""
    return pd.DataFrame(
        [
            {
                "Target": deployment.target,
                "Model": deployment.model_name,
                "Filter": deployment.filter_name,
                "Probability_Threshold": (
                    deployment.probability_threshold
                ),
                "Keep_Top_Fraction": (
                    deployment.keep_top_fraction
                ),
                "Model_Path": str(deployment.model_path),
            }
        ]
    )



def choose_model_ready_signal_date(
    featured_prices: pd.DataFrame,
    minimum_coverage_ratio: float = 0.60,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Choose the latest date with enough complete ML feature rows.

    This prevents the baseline portfolio from advancing to a stock date
    for which the market-derived ML features are not available yet.
    """
    if not 0 < minimum_coverage_ratio <= 1:
        raise ValueError(
            "minimum_coverage_ratio 0 ile 1 arasında olmalıdır."
        )

    required = {
        "Date",
        "Ticker",
    }.union(BASE_FEATURE_COLUMNS)

    missing = required.difference(featured_prices.columns)

    if missing:
        raise KeyError(
            "Model-ready tarih seçimi için eksik sütunlar: "
            f"{sorted(missing)}"
        )

    data = featured_prices.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    if as_of_date is not None:
        data = data.loc[
            data["Date"] <= pd.Timestamp(as_of_date)
        ]

    if data.empty:
        raise ValueError(
            "Model-ready sinyal tarihi için veri bulunamadı."
        )

    total_tickers = int(data["Ticker"].nunique())
    required_count = max(
        1,
        int(np.ceil(total_tickers * minimum_coverage_ratio)),
    )

    complete_mask = data[
        BASE_FEATURE_COLUMNS
    ].notna().all(axis=1)

    coverage = (
        data.loc[complete_mask]
        .groupby("Date")["Ticker"]
        .nunique()
        .sort_index()
    )

    valid_dates = coverage.loc[
        coverage >= required_count
    ]

    if valid_dates.empty:
        raise ValueError(
            "Yeterli sayıda tam ML özellikli hisse bulunan "
            "bir sinyal tarihi bulunamadı."
        )

    return pd.Timestamp(valid_dates.index.max())


def model_score_diagnostic(
    baseline_prices: pd.DataFrame,
    challenger_prices: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    deployment: ChallengerDeployment,
) -> pd.DataFrame:
    """Compare Robot AL rows with challenger probabilities and decisions."""
    selected_date = pd.Timestamp(signal_date)

    baseline_candidates = (
        baseline_prices.loc[
            pd.to_datetime(
                baseline_prices["Date"]
            ).eq(selected_date)
            & baseline_prices["Signal"].eq("AL"),
            [
                "Ticker",
                "Score",
                "RSI",
                "ADX",
                "RET_63",
                "RET_126",
            ],
        ]
        .drop_duplicates("Ticker", keep="last")
        .copy()
    )

    challenger_scores = (
        challenger_prices.loc[
            pd.to_datetime(
                challenger_prices["Date"]
            ).eq(selected_date),
            [
                "Ticker",
                "ML_Probability",
                "ML_Filter_Passed",
                "Signal",
            ],
        ]
        .rename(
            columns={
                "Signal": "Challenger_Signal",
            }
        )
        .drop_duplicates("Ticker", keep="last")
    )

    diagnostic = baseline_candidates.merge(
        challenger_scores,
        on="Ticker",
        how="left",
        validate="one_to_one",
    )

    diagnostic["Locked_Threshold"] = (
        deployment.probability_threshold
    )
    diagnostic["Distance_To_Threshold"] = (
        diagnostic["ML_Probability"]
        - diagnostic["Locked_Threshold"]
        if deployment.probability_threshold is not None
        else np.nan
    )

    diagnostic["ML_Decision"] = np.select(
        [
            diagnostic["ML_Probability"].isna(),
            diagnostic["ML_Filter_Passed"].eq(True),
        ],
        [
            "NO_SCORE",
            "PASS",
        ],
        default="REJECT",
    )

    return diagnostic.sort_values(
        "ML_Probability",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def assert_model_scores_available(
    diagnostic: pd.DataFrame,
) -> None:
    """Fail loudly instead of silently treating missing scores as rejects."""
    if diagnostic.empty:
        return

    missing_scores = diagnostic[
        "ML_Probability"
    ].isna()

    if missing_scores.any():
        tickers = diagnostic.loc[
            missing_scores,
            "Ticker",
        ].tolist()

        raise RuntimeError(
            "Seçilen model-ready tarihte bile bazı Robot AL "
            "sinyalleri skorlanamadı: "
            + ", ".join(tickers)
        )

def build_challenger_prices(
    featured_prices: pd.DataFrame,
    fitted_model: Any,
    deployment: ChallengerDeployment,
    prediction_start: str | pd.Timestamp,
    prediction_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Attach probabilities and apply the locked ML filter."""
    probability_prices = add_model_probabilities(
        featured_prices=featured_prices,
        fitted_model=fitted_model,
        feature_columns=BASE_FEATURE_COLUMNS,
        start=prediction_start,
        end=prediction_end,
    )

    filter_config = MLFilterConfig(
        name=deployment.filter_name,
        probability_threshold=(
            deployment.probability_threshold
        ),
        keep_top_fraction=(
            deployment.keep_top_fraction
        ),
    )

    return apply_ml_filter(
        probability_prices=probability_prices,
        filter_config=filter_config,
    )


def _plan_buy_view(
    plan: DailyPlan,
    portfolio_name: str,
) -> pd.DataFrame:
    if plan.buy_orders.empty:
        return pd.DataFrame(
            columns=[
                "Portfolio",
                "Ticker",
                "Rank",
                "Score",
                "Estimated_Entry",
                "Estimated_Stop",
                "Estimated_Shares",
                "ML_Probability",
            ]
        )

    frame = plan.buy_orders.copy()
    frame["Portfolio"] = portfolio_name

    if "ML_Probability" not in frame.columns:
        frame["ML_Probability"] = np.nan

    return frame[
        [
            "Portfolio",
            "Ticker",
            "Rank",
            "Score",
            "Estimated_Entry",
            "Estimated_Stop",
            "Estimated_Shares",
            "ML_Probability",
        ]
    ]


def _plan_sell_view(
    plan: DailyPlan,
    portfolio_name: str,
) -> pd.DataFrame:
    if plan.sell_orders.empty:
        return pd.DataFrame(
            columns=[
                "Portfolio",
                "Ticker",
                "Action",
                "Reason",
                "Close",
                "Stop_Loss",
                "Trailing_Level",
            ]
        )

    frame = plan.sell_orders.copy()
    frame["Portfolio"] = portfolio_name

    return frame[
        [
            "Portfolio",
            "Ticker",
            "Action",
            "Reason",
            "Close",
            "Stop_Loss",
            "Trailing_Level",
        ]
    ]


def create_dual_daily_plan(
    baseline_prices: pd.DataFrame,
    challenger_prices: pd.DataFrame,
    baseline_state: dict[str, Any],
    challenger_state: dict[str, Any],
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    signal_date: str | pd.Timestamp | None = None,
    minimum_coverage_ratio: float = 0.60,
) -> DualDailyPlan:
    """Create synchronized daily plans for both portfolios."""
    baseline_plan = create_daily_plan(
        scored_prices=baseline_prices,
        state=baseline_state,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
        signal_date=signal_date,
        minimum_coverage_ratio=minimum_coverage_ratio,
    )

    challenger_plan = create_daily_plan(
        scored_prices=challenger_prices,
        state=challenger_state,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
        signal_date=baseline_plan.signal_date,
        minimum_coverage_ratio=minimum_coverage_ratio,
    )

    # Attach the locked model score to challenger buy orders for audit.
    if not challenger_plan.buy_orders.empty:
        latest_probability_rows = (
            challenger_prices.loc[
                pd.to_datetime(challenger_prices["Date"]).eq(
                    baseline_plan.signal_date
                ),
                ["Ticker", "ML_Probability"],
            ]
            .drop_duplicates("Ticker", keep="last")
        )

        challenger_plan.buy_orders = (
            challenger_plan.buy_orders.merge(
                latest_probability_rows,
                on="Ticker",
                how="left",
                validate="many_to_one",
            )
        )

    buy_comparison = pd.concat(
        [
            _plan_buy_view(
                baseline_plan,
                "Baseline_Robot",
            ),
            _plan_buy_view(
                challenger_plan,
                "ML_Challenger",
            ),
        ],
        ignore_index=True,
    )

    if not buy_comparison.empty:
        counts = (
            buy_comparison.groupby("Ticker")["Portfolio"]
            .nunique()
            .rename("Portfolio_Count")
        )
        buy_comparison = buy_comparison.merge(
            counts,
            on="Ticker",
            how="left",
        )
        buy_comparison["Appears_In_Both"] = (
            buy_comparison["Portfolio_Count"].eq(2)
        )

    sell_comparison = pd.concat(
        [
            _plan_sell_view(
                baseline_plan,
                "Baseline_Robot",
            ),
            _plan_sell_view(
                challenger_plan,
                "ML_Challenger",
            ),
        ],
        ignore_index=True,
    )

    summary_comparison = pd.concat(
        [
            baseline_plan.summary.assign(
                Portfolio="Baseline_Robot"
            ),
            challenger_plan.summary.assign(
                Portfolio="ML_Challenger"
            ),
        ],
        ignore_index=True,
    )

    return DualDailyPlan(
        signal_date=baseline_plan.signal_date,
        baseline=baseline_plan,
        challenger=challenger_plan,
        buy_comparison=buy_comparison,
        sell_comparison=sell_comparison,
        summary_comparison=summary_comparison,
    )


def compare_states(
    baseline_state: dict[str, Any],
    challenger_state: dict[str, Any],
    latest_prices: dict[str, float],
) -> pd.DataFrame:
    """Mark both paper portfolios to market and compare them."""
    baseline = mark_to_market(
        baseline_state,
        latest_prices,
    )
    challenger = mark_to_market(
        challenger_state,
        latest_prices,
    )

    frame = pd.DataFrame(
        [
            {
                "Portfolio": "Baseline_Robot",
                **baseline,
            },
            {
                "Portfolio": "ML_Challenger",
                **challenger,
            },
        ]
    )

    baseline_equity = float(
        frame.loc[
            frame["Portfolio"].eq("Baseline_Robot"),
            "Equity",
        ].iloc[0]
    )

    frame["Equity_Difference_vs_Baseline_TL"] = (
        frame["Equity"] - baseline_equity
    )

    return frame


def append_dual_equity_snapshot(
    comparison: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    path: str | Path,
) -> Path:
    """Append both portfolio values to one dual-history CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = comparison.copy()
    frame.insert(
        0,
        "Signal_Date",
        pd.Timestamp(signal_date).isoformat(),
    )

    if output_path.exists():
        frame.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        frame.to_csv(
            output_path,
            index=False,
        )

    return output_path


def save_dual_plans(
    dual_plan: DualDailyPlan,
    output_root: str | Path,
) -> dict[str, Path]:
    """Save baseline, challenger and comparison plan files."""
    root = (
        Path(output_root)
        / dual_plan.signal_date.strftime("%Y-%m-%d")
    )
    root.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": root / "dual_summary.csv",
        "buy_comparison": root / "dual_buy_comparison.csv",
        "sell_comparison": root / "dual_sell_comparison.csv",
        "baseline_buys": root / "baseline_buy_orders.csv",
        "baseline_sells": root / "baseline_sell_orders.csv",
        "challenger_buys": root / "challenger_buy_orders.csv",
        "challenger_sells": root / "challenger_sell_orders.csv",
    }

    dual_plan.summary_comparison.to_csv(
        paths["summary"],
        index=False,
    )
    dual_plan.buy_comparison.to_csv(
        paths["buy_comparison"],
        index=False,
    )
    dual_plan.sell_comparison.to_csv(
        paths["sell_comparison"],
        index=False,
    )
    dual_plan.baseline.buy_orders.to_csv(
        paths["baseline_buys"],
        index=False,
    )
    dual_plan.baseline.sell_orders.to_csv(
        paths["baseline_sells"],
        index=False,
    )
    dual_plan.challenger.buy_orders.to_csv(
        paths["challenger_buys"],
        index=False,
    )
    dual_plan.challenger.sell_orders.to_csv(
        paths["challenger_sells"],
        index=False,
    )

    return paths


def record_buy_from_plan(
    state: dict[str, Any],
    buy_orders: pd.DataFrame,
    ticker: str,
    fill_date: str | pd.Timestamp,
    fill_price: float,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    current_equity: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record one actual paper fill using its plan row."""
    matching = buy_orders.loc[
        buy_orders["Ticker"].eq(ticker)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"{ticker} için tek bir alış planı bulunmalıdır."
        )

    row = matching.iloc[0]
    atr = float(row["ATR"])
    stop_loss = (
        float(fill_price)
        - strategy_config.initial_stop_atr * atr
    )

    shares = calculate_fill_position_size(
        state=state,
        fill_price=float(fill_price),
        stop_loss=stop_loss,
        current_equity=float(current_equity),
        portfolio_config=portfolio_config,
    )

    if shares <= 0:
        raise ValueError(
            f"{ticker} için hesaplanan lot sıfır."
        )

    record_buy(
        state=state,
        ticker=ticker,
        fill_date=fill_date,
        fill_price=float(fill_price),
        shares=shares,
        stop_loss=stop_loss,
        signal_score=int(row["Score"]),
        portfolio_config=portfolio_config,
    )

    fill_record = {
        "Ticker": ticker,
        "Fill_Date": pd.Timestamp(fill_date).isoformat(),
        "Fill_Price": float(fill_price),
        "ATR": atr,
        "Stop_Loss": stop_loss,
        "Shares": shares,
        "Signal_Score": int(row["Score"]),
    }

    return state, fill_record


def record_sell_from_plan(
    state: dict[str, Any],
    sell_orders: pd.DataFrame,
    ticker: str,
    fill_date: str | pd.Timestamp,
    fill_price: float,
    portfolio_config: PortfolioConfig,
    trades_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record one actual paper sell using its planned reason."""
    matching = sell_orders.loc[
        sell_orders["Ticker"].eq(ticker)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"{ticker} için tek bir satış planı bulunmalıdır."
        )

    row = matching.iloc[0]

    state, trade = record_sell(
        state=state,
        ticker=ticker,
        fill_date=fill_date,
        fill_price=float(fill_price),
        reason=str(row["Reason"]),
        portfolio_config=portfolio_config,
    )

    append_closed_trade(
        trade=trade,
        path=trades_path,
    )

    return state, trade


def persist_portfolio_state(
    state: dict[str, Any],
    state_path: str | Path,
) -> Path:
    """Persist one baseline or challenger state."""
    return save_paper_state(
        state=state,
        path=state_path,
    )
