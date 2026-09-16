"""Soft ML ranking and position-sizing experiments for RS126 Enhanced.

The module keeps every RS126 Enhanced AL signal eligible. ML probability is
used only for:

1. Candidate priority when portfolio slots are limited.
2. Per-trade risk-budget multipliers.

The existing Baseline, RS126 Enhanced, ML Challenger and hard-filter Hybrid
rules are not modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import PortfolioConfig, StrategyConfig
from src.metrics import portfolio_metrics


@dataclass(frozen=True)
class MLAllocationVariant:
    """One transparent soft-use configuration for ML probability."""

    name: str
    family: str
    priority_mode: str = "none"
    priority_strength: float = 0.0
    sizing_amplitude: float = 0.0

    def validate(self) -> None:
        valid_modes = {
            "none",
            "tie_break",
            "centered_blend",
        }

        if self.priority_mode not in valid_modes:
            raise ValueError(
                "priority_mode şu değerlerden biri olmalıdır: "
                f"{sorted(valid_modes)}"
            )

        if self.priority_strength < 0:
            raise ValueError(
                "priority_strength negatif olamaz."
            )

        if not 0 <= self.sizing_amplitude < 1:
            raise ValueError(
                "sizing_amplitude 0 dahil, 1 hariç olmalıdır."
            )


def reference_variant() -> MLAllocationVariant:
    return MLAllocationVariant(
        name="RS126_Enhanced_Reference",
        family="Reference",
    )


def default_allocation_variants() -> list[MLAllocationVariant]:
    """Return the small, pre-registered soft-allocation experiment grid."""
    return [
        reference_variant(),
        MLAllocationVariant(
            name="ML_TieBreak_Within_Score",
            family="ML_Ranking",
            priority_mode="tie_break",
            priority_strength=0.49,
        ),
        MLAllocationVariant(
            name="ML_Centered_Blend_1_00",
            family="ML_Ranking",
            priority_mode="centered_blend",
            priority_strength=1.00,
        ),
        MLAllocationVariant(
            name="ML_Size_80_120",
            family="ML_Position_Sizing",
            sizing_amplitude=0.20,
        ),
        MLAllocationVariant(
            name="ML_Size_65_135",
            family="ML_Position_Sizing",
            sizing_amplitude=0.35,
        ),
        MLAllocationVariant(
            name="ML_TieBreak_Size_80_120",
            family="ML_Ranking_And_Sizing",
            priority_mode="tie_break",
            priority_strength=0.49,
            sizing_amplitude=0.20,
        ),
    ]


def add_ml_allocation_columns(
    enhanced_probability_prices: pd.DataFrame,
    variant: MLAllocationVariant,
    probability_column: str = "ML_Probability",
) -> pd.DataFrame:
    """Add priority and risk multiplier without rejecting any Enhanced AL."""
    variant.validate()

    required = {
        "Date",
        "Ticker",
        "Score",
        "Signal",
        probability_column,
    }
    missing = required.difference(
        enhanced_probability_prices.columns
    )

    if missing:
        raise KeyError(
            "ML allocation deneyi için eksik sütunlar: "
            f"{sorted(missing)}"
        )

    result = enhanced_probability_prices.copy()
    result["Date"] = pd.to_datetime(
        result["Date"]
    )

    original_signal = result["Signal"].copy()
    enhanced_al = result["Signal"].eq("AL")

    missing_probability = (
        enhanced_al
        & result[probability_column].isna()
    )

    if missing_probability.any():
        examples = (
            result.loc[
                missing_probability,
                ["Date", "Ticker"],
            ]
            .head(20)
            .astype(str)
            .to_dict(orient="records")
        )

        raise RuntimeError(
            "RS126 Enhanced AL adaylarında ML olasılığı eksik. "
            f"İlk örnekler: {examples}"
        )

    result["ML_Daily_Quantile"] = np.nan
    result["ML_Daily_Centered"] = 0.0

    al_index = result.index[enhanced_al]

    if len(al_index):
        al_frame = result.loc[
            al_index,
            ["Date", probability_column],
        ].copy()

        daily_rank = (
            al_frame.groupby("Date")[
                probability_column
            ]
            .rank(
                method="average",
                ascending=True,
            )
        )
        daily_count = (
            al_frame.groupby("Date")[
                probability_column
            ]
            .transform("count")
        )

        # Midpoint empirical quantile:
        # one candidate -> 0.50, two candidates -> 0.25 / 0.75.
        quantile = (
            daily_rank - 0.5
        ) / daily_count

        result.loc[
            al_index,
            "ML_Daily_Quantile",
        ] = quantile
        result.loc[
            al_index,
            "ML_Daily_Centered",
        ] = (
            quantile * 2.0 - 1.0
        )

    base_score = pd.to_numeric(
        result["Score"],
        errors="coerce",
    )

    if variant.priority_mode == "none":
        priority_adjustment = pd.Series(
            0.0,
            index=result.index,
        )
    elif variant.priority_mode == "tie_break":
        # The adjustment remains below 0.50. Therefore an integer
        # Enhanced score of 12 always stays above a score of 11.
        priority_adjustment = (
            variant.priority_strength
            * result["ML_Daily_Quantile"]
            .fillna(0.0)
        )
    else:
        # A strength of 1.0 may allow an exceptionally strong lower-score
        # candidate to compete with a weak candidate one score point above.
        priority_adjustment = (
            variant.priority_strength
            * result["ML_Daily_Centered"]
            .fillna(0.0)
        )

    risk_multiplier = (
        1.0
        + variant.sizing_amplitude
        * result["ML_Daily_Centered"]
        .fillna(0.0)
    )

    lower_bound = (
        1.0 - variant.sizing_amplitude
    )
    upper_bound = (
        1.0 + variant.sizing_amplitude
    )

    risk_multiplier = risk_multiplier.clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    # Non-buy rows never create an entry, but keeping neutral values makes
    # diagnostics and reruns easier to interpret.
    priority_adjustment = (
        priority_adjustment.where(
            enhanced_al,
            0.0,
        )
    )
    risk_multiplier = (
        risk_multiplier.where(
            enhanced_al,
            1.0,
        )
    )

    result["Priority_Adjustment"] = (
        priority_adjustment
    )
    result["Priority_Score"] = (
        base_score
        + priority_adjustment
    )
    result["Risk_Multiplier"] = (
        risk_multiplier
    )
    result["ML_Allocation_Variant"] = (
        variant.name
    )

    if not result["Signal"].equals(
        original_signal
    ):
        raise RuntimeError(
            "Soft allocation deneyi Signal sütununu değiştirdi."
        )

    return result


def _next_bar_date(
    frame: pd.DataFrame,
    current_position: int,
) -> pd.Timestamp | None:
    next_position = current_position + 1

    if next_position >= len(frame):
        return None

    return pd.Timestamp(
        frame.iloc[next_position]["Date"]
    )


def _return_and_outlier(
    entry_price: float,
    exit_price: float,
    portfolio_config: PortfolioConfig,
) -> tuple[float, bool]:
    ret = (
        exit_price
        * (
            1
            - portfolio_config.commission_rate
        )
        / (
            entry_price
            * (
                1
                + portfolio_config.commission_rate
            )
        )
        - 1
    )

    is_outlier = not (
        portfolio_config.minimum_valid_return
        <= ret
        <= portfolio_config.maximum_valid_return
    )

    return ret, is_outlier


def _rank_allocation_candidates(
    daily_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Rank AL rows using the experiment's priority before Robot tie-breaks."""
    candidates = daily_rows.loc[
        daily_rows["Signal"].eq("AL")
    ].copy()

    required = {
        "Priority_Score",
        "Score",
        "RET_126",
        "RET_63",
        "ADX",
    }
    missing = required.difference(
        candidates.columns
    )

    if missing:
        raise KeyError(
            "Allocation sıralaması için eksik sütunlar: "
            f"{sorted(missing)}"
        )

    return candidates.sort_values(
        [
            "Priority_Score",
            "Score",
            "RET_126",
            "RET_63",
            "ADX",
        ],
        ascending=False,
    ).reset_index(drop=True)


def run_allocation_backtest(
    allocated_prices: pd.DataFrame,
    strategy_config: StrategyConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Robot with optional ML priority and per-entry risk multipliers."""
    strategy_config = (
        strategy_config
        or StrategyConfig()
    )
    portfolio_config = (
        portfolio_config
        or PortfolioConfig()
    )

    required = {
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "LOW_10_PREV",
        "RET_63",
        "RET_126",
        "ADX",
        "RSI",
        "Score",
        "Signal",
        "Priority_Score",
        "Risk_Multiplier",
        "ML_Probability",
    }
    missing = required.difference(
        allocated_prices.columns
    )

    if missing:
        raise KeyError(
            "Allocation backtest için eksik sütunlar: "
            f"{sorted(missing)}"
        )

    data_dict: dict[
        str,
        pd.DataFrame,
    ] = {}
    date_position: dict[
        str,
        dict[pd.Timestamp, int],
    ] = {}

    for ticker, ticker_df in (
        allocated_prices.groupby(
            "Ticker",
            sort=False,
        )
    ):
        work = (
            ticker_df.sort_values("Date")
            .drop_duplicates(
                "Date",
                keep="last",
            )
            .reset_index(drop=True)
        )

        data_dict[ticker] = work
        date_position[ticker] = {
            pd.Timestamp(date): position
            for position, date in enumerate(
                work["Date"]
            )
        }

    all_dates = sorted(
        set().union(
            *[
                set(frame["Date"])
                for frame in data_dict.values()
            ]
        )
    )

    if not all_dates:
        raise ValueError(
            "Allocation backtest için tarih bulunamadı."
        )

    cash = portfolio_config.initial_capital
    positions: dict[
        str,
        dict[str, Any],
    ] = {}
    pending_entries: list[
        dict[str, Any]
    ] = []
    trades: list[
        dict[str, Any]
    ] = []
    equity_curve: list[
        dict[str, Any]
    ] = []

    def current_equity(
        date: pd.Timestamp,
        use_open: bool = False,
    ) -> float:
        value = cash

        for symbol, position in (
            positions.items()
        ):
            frame = data_dict[symbol]
            position_index = (
                date_position[symbol].get(
                    date
                )
            )

            if position_index is not None:
                column = (
                    "Open"
                    if use_open
                    else "Close"
                )
                mark = float(
                    frame.iloc[
                        position_index
                    ][column]
                )
            else:
                mark = float(
                    position["last_price"]
                )

            value += (
                int(position["shares"])
                * mark
            )

        return float(value)

    def close_position(
        symbol: str,
        date: pd.Timestamp,
        exit_price: float,
        reason: str,
    ) -> None:
        nonlocal cash

        position = positions[symbol]
        net_value = (
            position["shares"]
            * exit_price
            * (
                1
                - portfolio_config.commission_rate
            )
        )
        cash += net_value

        ret, is_outlier = (
            _return_and_outlier(
                position["entry_price"],
                exit_price,
                portfolio_config,
            )
        )

        trades.append(
            {
                "Ticker": symbol,
                "Entry_Date": (
                    position["entry_date"]
                ),
                "Exit_Date": date,
                "Entry": (
                    position["entry_price"]
                ),
                "Exit": exit_price,
                "Shares": position["shares"],
                "Return": ret,
                "Return_%": ret * 100,
                "Reason": reason,
                "Is_Outlier": is_outlier,
                "Signal_Score": (
                    position["signal_score"]
                ),
                "Priority_Score": (
                    position["priority_score"]
                ),
                "ML_Probability": (
                    position["ml_probability"]
                ),
                "Risk_Multiplier": (
                    position["risk_multiplier"]
                ),
            }
        )

        del positions[symbol]

    for current_date in all_dates:
        current_date = pd.Timestamp(
            current_date
        )

        # 1. Prior close-based exits execute at today's open.
        for symbol in list(positions):
            position = positions[symbol]

            if not position.get(
                "pending_exit_reason"
            ):
                continue

            position_index = (
                date_position[symbol].get(
                    current_date
                )
            )

            if (
                position_index is None
                or current_date
                <= position[
                    "pending_exit_trigger_date"
                ]
            ):
                continue

            row = data_dict[symbol].iloc[
                position_index
            ]
            exit_price = float(
                row["Open"]
            ) * (
                1
                - portfolio_config.slippage_rate
            )

            close_position(
                symbol,
                current_date,
                exit_price,
                str(
                    position[
                        "pending_exit_reason"
                    ]
                ),
            )

        # 2. Pending entries execute at today's open.
        todays_entries = [
            order
            for order in pending_entries
            if pd.Timestamp(
                order["Entry_Date"]
            )
            == current_date
        ]
        pending_entries = [
            order
            for order in pending_entries
            if pd.Timestamp(
                order["Entry_Date"]
            )
            != current_date
        ]

        for order in todays_entries:
            symbol = str(
                order["Ticker"]
            )

            if symbol in positions:
                continue
            if (
                len(positions)
                >= portfolio_config.max_positions
            ):
                continue

            position_index = (
                date_position[symbol].get(
                    current_date
                )
            )

            if position_index is None:
                continue

            row = data_dict[symbol].iloc[
                position_index
            ]
            raw_open = float(row["Open"])
            entry_price = raw_open * (
                1
                + portfolio_config.slippage_rate
            )
            stop_distance = (
                strategy_config.initial_stop_atr
                * float(
                    order["Signal_ATR"]
                )
            )

            if (
                entry_price <= 0
                or stop_distance <= 0
            ):
                continue

            stop_loss = (
                entry_price
                - stop_distance
            )

            if stop_loss <= 0:
                continue

            equity_at_open = current_equity(
                current_date,
                use_open=True,
            )

            risk_multiplier = float(
                order["Risk_Multiplier"]
            )

            if risk_multiplier <= 0:
                continue

            risk_amount = (
                equity_at_open
                * portfolio_config.risk_per_trade
                * risk_multiplier
            )

            estimated_stop_exit = (
                stop_loss
                * (
                    1
                    - portfolio_config.slippage_rate
                )
            )

            risk_per_share = (
                entry_price
                * (
                    1
                    + portfolio_config.commission_rate
                )
                - estimated_stop_exit
                * (
                    1
                    - portfolio_config.commission_rate
                )
            )

            if risk_per_share <= 0:
                continue

            shares_by_risk = int(
                risk_amount
                / risk_per_share
            )

            cost_per_share = (
                entry_price
                * (
                    1
                    + portfolio_config.commission_rate
                )
            )
            shares_by_cash = int(
                cash
                / cost_per_share
            )

            if (
                portfolio_config.max_position_fraction
                is None
            ):
                shares_by_position_cap = (
                    shares_by_cash
                )
            else:
                if not (
                    0
                    < portfolio_config.max_position_fraction
                    <= 1
                ):
                    raise ValueError(
                        "max_position_fraction 0 ile 1 arasında "
                        "olmalı veya None olmalıdır."
                    )

                position_budget = (
                    equity_at_open
                    * portfolio_config.max_position_fraction
                )
                shares_by_position_cap = int(
                    position_budget
                    / cost_per_share
                )

            shares = min(
                shares_by_risk,
                shares_by_cash,
                shares_by_position_cap,
            )

            if shares <= 0:
                continue

            cost = (
                shares
                * entry_price
                * (
                    1
                    + portfolio_config.commission_rate
                )
            )
            cash -= cost

            positions[symbol] = {
                "entry_date": current_date,
                "entry_price": entry_price,
                "shares": shares,
                "stop_loss": stop_loss,
                "highest_price": entry_price,
                "last_price": entry_price,
                "signal_score": int(
                    round(
                        float(
                            order[
                                "Signal_Score"
                            ]
                        )
                    )
                ),
                "priority_score": float(
                    order[
                        "Priority_Score"
                    ]
                ),
                "ml_probability": float(
                    order[
                        "ML_Probability"
                    ]
                ),
                "risk_multiplier": (
                    risk_multiplier
                ),
                "pending_exit_reason": None,
                "pending_exit_trigger_date": None,
            }

        # 3. Intraday stops and close-based exits.
        for symbol in list(positions):
            position = positions[symbol]
            position_index = (
                date_position[symbol].get(
                    current_date
                )
            )

            if position_index is None:
                continue

            row = data_dict[symbol].iloc[
                position_index
            ]
            open_price = float(
                row["Open"]
            )
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            atr = float(row["ATR"])
            low_10_prev = float(
                row["LOW_10_PREV"]
            )

            position["last_price"] = close
            position["highest_price"] = max(
                float(
                    position["highest_price"]
                ),
                high,
            )

            if (
                open_price
                <= position["stop_loss"]
            ):
                exit_price = (
                    open_price
                    * (
                        1
                        - portfolio_config.slippage_rate
                    )
                )
                close_position(
                    symbol,
                    current_date,
                    exit_price,
                    "Stop Loss Gap",
                )
                continue

            if (
                low
                <= position["stop_loss"]
            ):
                exit_price = (
                    position["stop_loss"]
                    * (
                        1
                        - portfolio_config.slippage_rate
                    )
                )
                close_position(
                    symbol,
                    current_date,
                    exit_price,
                    "Stop Loss",
                )
                continue

            trailing_stop = (
                position["highest_price"]
                - strategy_config.trailing_stop_atr
                * atr
            )
            profit_pct = (
                close
                - position["entry_price"]
            ) / position["entry_price"]

            if (
                profit_pct
                > strategy_config.trailing_activation_return
                and close < trailing_stop
            ):
                position[
                    "pending_exit_reason"
                ] = "Trailing Stop"
                position[
                    "pending_exit_trigger_date"
                ] = current_date
            elif close < low_10_prev:
                position[
                    "pending_exit_reason"
                ] = "LOW10 Altı"
                position[
                    "pending_exit_trigger_date"
                ] = current_date

        # 4. Generate signals and schedule next-open entries.
        pending_symbols = {
            str(order["Ticker"])
            for order in pending_entries
        }
        expected_exits = sum(
            bool(
                position.get(
                    "pending_exit_reason"
                )
            )
            for position in positions.values()
        )
        available_slots = (
            portfolio_config.max_positions
            - (
                len(positions)
                - expected_exits
            )
            - len(pending_entries)
        )

        if available_slots > 0:
            daily_rows = []

            for symbol, frame in (
                data_dict.items()
            ):
                if (
                    symbol in positions
                    or symbol in pending_symbols
                ):
                    continue

                position_index = (
                    date_position[symbol].get(
                        current_date
                    )
                )

                if position_index is None:
                    continue

                next_date = _next_bar_date(
                    frame,
                    position_index,
                )

                if next_date is None:
                    continue

                daily_rows.append(
                    frame.iloc[
                        position_index
                    ]
                )

            if daily_rows:
                daily_frame = pd.DataFrame(
                    daily_rows
                )
                candidates = (
                    _rank_allocation_candidates(
                        daily_frame
                    )
                )

                for _, row in (
                    candidates.head(
                        available_slots
                    ).iterrows()
                ):
                    symbol = str(
                        row["Ticker"]
                    )
                    position_index = (
                        date_position[symbol][
                            current_date
                        ]
                    )
                    next_date = _next_bar_date(
                        data_dict[symbol],
                        position_index,
                    )

                    if next_date is None:
                        continue

                    pending_entries.append(
                        {
                            "Ticker": symbol,
                            "Signal_Date": (
                                current_date
                            ),
                            "Entry_Date": next_date,
                            "Signal_Score": float(
                                row["Score"]
                            ),
                            "Priority_Score": float(
                                row[
                                    "Priority_Score"
                                ]
                            ),
                            "Risk_Multiplier": float(
                                row[
                                    "Risk_Multiplier"
                                ]
                            ),
                            "ML_Probability": float(
                                row[
                                    "ML_Probability"
                                ]
                            ),
                            "Signal_ATR": float(
                                row["ATR"]
                            ),
                        }
                    )

        equity_curve.append(
            {
                "Date": current_date,
                "Equity": current_equity(
                    current_date,
                    use_open=False,
                ),
                "Cash": cash,
                "Open_Positions": (
                    len(positions)
                ),
                "Pending_Entries": (
                    len(pending_entries)
                ),
            }
        )

    final_date = pd.Timestamp(
        all_dates[-1]
    )

    for symbol in list(positions):
        position = positions[symbol]
        exit_price = float(
            position["last_price"]
        ) * (
            1
            - portfolio_config.slippage_rate
        )
        close_position(
            symbol,
            final_date,
            exit_price,
            "Final Close",
        )

    if equity_curve:
        equity_curve[-1]["Equity"] = cash
        equity_curve[-1]["Cash"] = cash
        equity_curve[-1][
            "Open_Positions"
        ] = 0

    equity = pd.DataFrame(
        equity_curve
    )
    trades_frame = pd.DataFrame(
        trades
    )

    return equity, trades_frame


def evaluate_allocation_variant(
    enhanced_probability_prices: pd.DataFrame,
    variant: MLAllocationVariant,
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Backtest one ranking/sizing variant in the common date window."""
    period = enhanced_probability_prices.loc[
        pd.to_datetime(
            enhanced_probability_prices[
                "Date"
            ]
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

    allocated = add_ml_allocation_columns(
        enhanced_probability_prices=period,
        variant=variant,
    )

    equity, trades = (
        run_allocation_backtest(
            allocated_prices=allocated,
            strategy_config=strategy_config,
            portfolio_config=portfolio_config,
        )
    )

    metrics = portfolio_metrics(
        equity,
        trades,
    )

    al_rows = allocated.loc[
        allocated["Signal"].eq("AL")
    ]

    actual_risk = (
        pd.to_numeric(
            trades.get(
                "Risk_Multiplier",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
    )
    actual_probability = (
        pd.to_numeric(
            trades.get(
                "ML_Probability",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
    )

    metrics.update(
        {
            "Variant": variant.name,
            "Family": variant.family,
            "Priority_Mode": (
                variant.priority_mode
            ),
            "Priority_Strength": (
                variant.priority_strength
            ),
            "Sizing_Amplitude": (
                variant.sizing_amplitude
            ),
            "Eligible_AL_Rows": int(
                len(al_rows)
            ),
            "Signal_Change_Count": 0,
            "Candidate_Mean_Risk_Multiplier": (
                float(
                    al_rows[
                        "Risk_Multiplier"
                    ].mean()
                )
                if len(al_rows)
                else np.nan
            ),
            "Entered_Mean_Risk_Multiplier": (
                float(actual_risk.mean())
                if len(actual_risk)
                else np.nan
            ),
            "Entered_Max_Risk_Multiplier": (
                float(actual_risk.max())
                if len(actual_risk)
                else np.nan
            ),
            "Entered_Mean_ML_Probability": (
                float(
                    actual_probability.mean()
                )
                if len(actual_probability)
                else np.nan
            ),
            "Period_Start": (
                pd.Timestamp(start)
            ),
            "Period_End": (
                pd.Timestamp(end)
            ),
        }
    )

    return metrics, equity, trades


def run_allocation_grid(
    enhanced_probability_prices: pd.DataFrame,
    variants: Iterable[
        MLAllocationVariant
    ],
    strategy_config: StrategyConfig,
    portfolio_config: PortfolioConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    show_progress: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, pd.DataFrame]],
]:
    """Evaluate every pre-registered soft-use variant."""
    variants = list(variants)

    iterator: Iterable[
        MLAllocationVariant
    ] = variants

    if show_progress:
        iterator = tqdm(
            variants,
            desc=(
                "RS126 soft ML allocation experiments"
            ),
        )

    records: list[
        dict[str, Any]
    ] = []
    outputs: dict[
        str,
        dict[str, pd.DataFrame],
    ] = {}

    for variant in iterator:
        try:
            metrics, equity, trades = (
                evaluate_allocation_variant(
                    enhanced_probability_prices=(
                        enhanced_probability_prices
                    ),
                    variant=variant,
                    strategy_config=(
                        strategy_config
                    ),
                    portfolio_config=(
                        portfolio_config
                    ),
                    start=start,
                    end=end,
                )
            )
            metrics["Status"] = "OK"
            metrics["Error"] = ""

            outputs[variant.name] = {
                "equity": equity,
                "trades": trades,
            }

        except Exception as error:
            metrics = {
                "Variant": variant.name,
                "Family": variant.family,
                "Status": "ERROR",
                "Error": str(error),
                **{
                    f"Variant_{key}": value
                    for key, value in asdict(
                        variant
                    ).items()
                },
            }

        records.append(metrics)

    return (
        pd.DataFrame(records),
        outputs,
    )


def reference_parity_table(
    allocation_results: pd.DataFrame,
    reference_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Confirm custom reference reproduces the existing RS126 backtest."""
    custom_rows = allocation_results.loc[
        allocation_results["Variant"].eq(
            "RS126_Enhanced_Reference"
        )
        & allocation_results[
            "Status"
        ].eq("OK")
    ]
    reference_rows = reference_metrics.loc[
        reference_metrics[
            "Portfolio"
        ].eq("RS126_Enhanced")
    ]

    if (
        len(custom_rows) != 1
        or len(reference_rows) != 1
    ):
        raise ValueError(
            "Reference parity için tek satır bulunmalıdır."
        )

    custom = custom_rows.iloc[0]
    reference = reference_rows.iloc[0]

    columns = [
        "End_Value",
        "CAGR_%",
        "Max_Drawdown_%",
        "Profit_Factor",
        "Sharpe",
        "Calmar",
        "Trade_Count",
    ]

    records = []

    for column in columns:
        custom_value = pd.to_numeric(
            pd.Series(
                [custom.get(column)]
            ),
            errors="coerce",
        ).iloc[0]
        reference_value = pd.to_numeric(
            pd.Series(
                [reference.get(column)]
            ),
            errors="coerce",
        ).iloc[0]

        records.append(
            {
                "Metric": column,
                "Existing_RS126": (
                    reference_value
                ),
                "Custom_Reference": (
                    custom_value
                ),
                "Difference": (
                    custom_value
                    - reference_value
                ),
            }
        )

    return pd.DataFrame(records)


def allocation_pairwise_table(
    allocation_results: pd.DataFrame,
    reference_variant_name: str = (
        "RS126_Enhanced_Reference"
    ),
) -> pd.DataFrame:
    """Compare every soft-use candidate with unchanged RS126 Enhanced."""
    successful = allocation_results.loc[
        allocation_results[
            "Status"
        ].eq("OK")
    ].copy()

    reference_rows = successful.loc[
        successful["Variant"].eq(
            reference_variant_name
        )
    ]

    if len(reference_rows) != 1:
        raise ValueError(
            "Allocation karşılaştırması için tek reference gerekir."
        )

    reference = reference_rows.iloc[0]
    records = []

    for _, candidate in successful.iterrows():
        if (
            candidate["Variant"]
            == reference_variant_name
        ):
            continue

        trade_fraction = (
            candidate["Trade_Count"]
            / reference["Trade_Count"]
            if reference["Trade_Count"]
            else np.nan
        )
        calmar_improvement = (
            (
                candidate["Calmar"]
                / abs(reference["Calmar"])
                - 1
            )
            * 100
            if (
                pd.notna(
                    reference["Calmar"]
                )
                and abs(
                    reference["Calmar"]
                )
                > 1e-12
            )
            else np.nan
        )

        records.append(
            {
                "Variant": (
                    candidate["Variant"]
                ),
                "Family": (
                    candidate["Family"]
                ),
                "End_Value_Difference_TL": (
                    candidate["End_Value"]
                    - reference["End_Value"]
                ),
                "CAGR_Difference_pp": (
                    candidate["CAGR_%"]
                    - reference["CAGR_%"]
                ),
                "Max_DD_Improvement_pp": (
                    candidate[
                        "Max_Drawdown_%"
                    ]
                    - reference[
                        "Max_Drawdown_%"
                    ]
                ),
                "Profit_Factor_Difference": (
                    candidate[
                        "Profit_Factor"
                    ]
                    - reference[
                        "Profit_Factor"
                    ]
                ),
                "Sharpe_Difference": (
                    candidate["Sharpe"]
                    - reference["Sharpe"]
                ),
                "Calmar_Difference": (
                    candidate["Calmar"]
                    - reference["Calmar"]
                ),
                "Calmar_Improvement_%": (
                    calmar_improvement
                ),
                "Trade_Fraction": (
                    trade_fraction
                ),
                "Exposure_Difference_pp": (
                    candidate["Exposure_%"]
                    - reference["Exposure_%"]
                ),
                "Entered_Mean_Risk_Multiplier": (
                    candidate[
                        "Entered_Mean_Risk_Multiplier"
                    ]
                ),
                "Entered_Max_Risk_Multiplier": (
                    candidate[
                        "Entered_Max_Risk_Multiplier"
                    ]
                ),
            }
        )

    return pd.DataFrame(records)


def allocation_historical_screen(
    pairwise: pd.DataFrame,
    minimum_trade_fraction: float = 0.90,
    minimum_return_case_cagr_pp: float = 1.50,
    maximum_return_case_dd_deterioration_pp: float = 1.00,
    minimum_return_case_calmar_pct: float = 5.0,
    minimum_risk_case_cagr_pp: float = -0.50,
    minimum_risk_case_dd_improvement_pp: float = 1.50,
    minimum_risk_case_calmar_pct: float = 8.0,
    minimum_profit_factor_ratio: float = 0.95,
) -> pd.DataFrame:
    """Apply a predeclared screen; passing means forward-test candidate only."""
    if pairwise.empty:
        return pairwise.copy()

    table = pairwise.copy()

    reference_pf = (
        table[
            "Profit_Factor_Difference"
        ]
        * 0
        + np.nan
    )

    # The pairwise table stores a difference, so the risk case uses the
    # conservative direct condition below instead of reconstructing a ratio.
    table["Return_Case_Passed"] = (
        table["CAGR_Difference_pp"].ge(
            minimum_return_case_cagr_pp
        )
        & table[
            "Max_DD_Improvement_pp"
        ].ge(
            -maximum_return_case_dd_deterioration_pp
        )
        & table[
            "Calmar_Improvement_%"
        ].ge(
            minimum_return_case_calmar_pct
        )
        & table[
            "Profit_Factor_Difference"
        ].ge(0.0)
        & table["Trade_Fraction"].ge(
            minimum_trade_fraction
        )
    )

    table["Risk_Case_Passed"] = (
        table["CAGR_Difference_pp"].ge(
            minimum_risk_case_cagr_pp
        )
        & table[
            "Max_DD_Improvement_pp"
        ].ge(
            minimum_risk_case_dd_improvement_pp
        )
        & table[
            "Calmar_Improvement_%"
        ].ge(
            minimum_risk_case_calmar_pct
        )
        & table[
            "Profit_Factor_Difference"
        ].ge(-0.10)
        & table["Trade_Fraction"].ge(
            minimum_trade_fraction
        )
    )

    table[
        "Historical_Screen_Passed"
    ] = (
        table["Return_Case_Passed"]
        | table["Risk_Case_Passed"]
    )
    table["Independent_Audit"] = False
    table["Decision"] = np.where(
        table[
            "Historical_Screen_Passed"
        ],
        "İleri dönem paper-trading adayı",
        "RS126 Enhanced korunur",
    )

    return table.sort_values(
        [
            "Historical_Screen_Passed",
            "Calmar_Improvement_%",
            "CAGR_Difference_pp",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)


def combine_equity_curves(
    reference_equity: pd.DataFrame,
    allocation_outputs: Mapping[
        str,
        Mapping[str, pd.DataFrame],
    ],
) -> pd.DataFrame:
    """Add soft-use equity columns to the existing comparison curve."""
    combined = reference_equity.copy()
    combined["Date"] = pd.to_datetime(
        combined["Date"]
    )

    for variant_name, output in (
        allocation_outputs.items()
    ):
        if (
            variant_name
            == "RS126_Enhanced_Reference"
        ):
            continue

        equity = (
            output["equity"][
                ["Date", "Equity"]
            ]
            .copy()
            .assign(
                Date=lambda frame: (
                    pd.to_datetime(
                        frame["Date"]
                    )
                )
            )
            .rename(
                columns={
                    "Equity": variant_name,
                }
            )
        )

        combined = combined.merge(
            equity,
            on="Date",
            how="inner",
            validate="one_to_one",
        )

    return combined.sort_values(
        "Date"
    ).reset_index(drop=True)


def yearly_return_table(
    combined_equity: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate calendar returns for every equity column."""
    data = combined_equity.copy()
    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    value_columns = [
        column
        for column in data.columns
        if column != "Date"
    ]

    records = []

    for year, group in data.groupby(
        data["Date"].dt.year
    ):
        row: dict[str, Any] = {
            "Year": int(year),
        }

        for column in value_columns:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            ).dropna()

            row[column] = (
                (
                    values.iloc[-1]
                    / values.iloc[0]
                    - 1
                )
                * 100
                if len(values) >= 2
                else np.nan
            )

        records.append(row)

    return pd.DataFrame(records)


def save_allocation_artifacts(
    output_directory: str | Path,
    allocation_results: pd.DataFrame,
    pairwise: pd.DataFrame,
    screen: pd.DataFrame,
    parity: pd.DataFrame,
    combined_equity: pd.DataFrame,
    yearly: pd.DataFrame,
    allocation_outputs: Mapping[
        str,
        Mapping[str, pd.DataFrame],
    ],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Save local experiment artifacts under the ignored results directory."""
    output = Path(output_directory)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "metrics": (
            output
            / "allocation_metrics.csv"
        ),
        "pairwise": (
            output
            / "allocation_pairwise.csv"
        ),
        "screen": (
            output
            / "allocation_historical_screen.csv"
        ),
        "parity": (
            output
            / "allocation_reference_parity.csv"
        ),
        "equity": (
            output
            / "allocation_equity.parquet"
        ),
        "yearly": (
            output
            / "allocation_yearly.csv"
        ),
        "trades": (
            output
            / "allocation_trades.parquet"
        ),
        "metadata": (
            output
            / "allocation_metadata.json"
        ),
    }

    allocation_results.to_csv(
        paths["metrics"],
        index=False,
    )
    pairwise.to_csv(
        paths["pairwise"],
        index=False,
    )
    screen.to_csv(
        paths["screen"],
        index=False,
    )
    parity.to_csv(
        paths["parity"],
        index=False,
    )
    combined_equity.to_parquet(
        paths["equity"],
        index=False,
    )
    yearly.to_csv(
        paths["yearly"],
        index=False,
    )

    trade_frames = []

    for variant_name, output_data in (
        allocation_outputs.items()
    ):
        trades = output_data[
            "trades"
        ].copy()
        trades["Variant"] = (
            variant_name
        )
        trade_frames.append(trades)

    pd.concat(
        trade_frames,
        ignore_index=True,
        sort=False,
    ).to_parquet(
        paths["trades"],
        index=False,
    )

    with paths["metadata"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dict(metadata),
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return paths
