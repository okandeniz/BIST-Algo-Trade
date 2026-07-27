"""BIST100 benchmark construction and active-performance metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def _prepare_market_prices(
    market_prices: pd.DataFrame,
) -> pd.DataFrame:
    required = {"Date", "Open", "Close"}
    missing = required.difference(market_prices.columns)

    if missing:
        raise KeyError(
            f"Benchmark için eksik sütunlar: {sorted(missing)}"
        )

    market = market_prices.copy()
    market["Date"] = pd.to_datetime(market["Date"])

    return (
        market.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def build_benchmark_equity(
    market_prices: pd.DataFrame,
    comparison_dates: Iterable[pd.Timestamp],
    initial_capital: float,
    commission_rate: float = 0.0,
    slippage_rate: float = 0.0,
    include_costs: bool = False,
    benchmark_name: str = "BIST100",
) -> pd.DataFrame:
    """Build a BIST100 buy-and-hold equity curve.

    Gross mode normalizes the index close to initial_capital and applies
    no transaction costs.

    Net mode buys at the first available open, includes entry costs and
    liquidates at the last close after exit costs.
    """
    if initial_capital <= 0:
        raise ValueError(
            "initial_capital sıfırdan büyük olmalıdır."
        )

    if not 0 <= commission_rate < 1:
        raise ValueError(
            "commission_rate 0 ile 1 arasında olmalıdır."
        )

    if not 0 <= slippage_rate < 1:
        raise ValueError(
            "slippage_rate 0 ile 1 arasında olmalıdır."
        )

    market = _prepare_market_prices(market_prices)

    dates = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                pd.Index(comparison_dates).drop_duplicates()
            )
        }
    ).sort_values("Date")

    aligned = dates.merge(
        market[["Date", "Open", "Close"]],
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    if len(aligned) < 2:
        raise ValueError(
            "Benchmark oluşturmak için en az iki ortak tarih gerekir."
        )

    if include_costs:
        entry_price = float(aligned.iloc[0]["Open"]) * (
            1 + slippage_rate
        )
        entry_cost_per_unit = entry_price * (
            1 + commission_rate
        )
        units = int(initial_capital / entry_cost_per_unit)

        if units <= 0:
            raise ValueError(
                "Başlangıç sermayesi benchmark alımı için yetersiz."
            )

        cash = (
            initial_capital
            - units * entry_cost_per_unit
        )

        aligned["Equity"] = (
            cash + units * aligned["Close"]
        )
        aligned["Cash"] = cash
        aligned["Open_Positions"] = 1

        final_exit_price = float(
            aligned.iloc[-1]["Close"]
        ) * (1 - slippage_rate)

        final_equity = (
            cash
            + units
            * final_exit_price
            * (1 - commission_rate)
        )

        aligned.loc[
            aligned.index[-1],
            ["Equity", "Cash", "Open_Positions"],
        ] = [final_equity, final_equity, 0]

        aligned["Units"] = units
        aligned["Entry_Price"] = entry_price
        mode = "Net"

    else:
        first_close = float(aligned.iloc[0]["Close"])

        if first_close <= 0:
            raise ValueError(
                "İlk benchmark kapanışı sıfırdan büyük olmalıdır."
            )

        aligned["Equity"] = (
            initial_capital
            * aligned["Close"]
            / first_close
        )
        aligned["Cash"] = 0.0
        aligned["Open_Positions"] = 1
        aligned["Units"] = np.nan
        aligned["Entry_Price"] = first_close
        mode = "Gross"

    aligned["Pending_Entries"] = 0
    aligned["Benchmark"] = benchmark_name
    aligned["Benchmark_Mode"] = mode

    return aligned[
        [
            "Date",
            "Equity",
            "Cash",
            "Open_Positions",
            "Pending_Entries",
            "Benchmark",
            "Benchmark_Mode",
            "Open",
            "Close",
            "Units",
            "Entry_Price",
        ]
    ].reset_index(drop=True)


def align_equity_curves(
    strategy_equity: pd.DataFrame,
    benchmark_equity: pd.DataFrame,
) -> pd.DataFrame:
    """Align strategy and benchmark equity on common dates."""
    required = {"Date", "Equity"}

    for name, frame in {
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
    }.items():
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(
                f"{name} için eksik sütunlar: {sorted(missing)}"
            )

    strategy = strategy_equity[
        ["Date", "Equity"]
    ].copy()
    benchmark = benchmark_equity[
        ["Date", "Equity"]
    ].copy()

    strategy["Date"] = pd.to_datetime(strategy["Date"])
    benchmark["Date"] = pd.to_datetime(benchmark["Date"])

    strategy = strategy.rename(
        columns={"Equity": "Strategy_Equity"}
    )
    benchmark = benchmark.rename(
        columns={"Equity": "Benchmark_Equity"}
    )

    return (
        strategy.merge(
            benchmark,
            on="Date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )


def active_performance_metrics(
    aligned_equity: pd.DataFrame,
    annualization_factor: int = 252,
) -> dict[str, float]:
    """Calculate alpha, beta and active-return diagnostics."""
    required = {
        "Date",
        "Strategy_Equity",
        "Benchmark_Equity",
    }
    missing = required.difference(aligned_equity.columns)

    if missing:
        raise KeyError(
            f"Aktif metrikler için eksik sütunlar: {sorted(missing)}"
        )

    data = aligned_equity.sort_values("Date").copy()
    returns = data[
        ["Strategy_Equity", "Benchmark_Equity"]
    ].pct_change(fill_method=None).dropna()

    returns.columns = [
        "Strategy_Return",
        "Benchmark_Return",
    ]

    if returns.empty:
        raise ValueError(
            "Aktif performans metrikleri için getiri bulunmuyor."
        )

    strategy_return = returns["Strategy_Return"]
    benchmark_return = returns["Benchmark_Return"]
    excess_return = strategy_return - benchmark_return

    benchmark_variance = benchmark_return.var(ddof=1)

    beta = (
        strategy_return.cov(benchmark_return)
        / benchmark_variance
        if benchmark_variance > 0
        else np.nan
    )

    alpha_daily = (
        strategy_return.mean()
        - beta * benchmark_return.mean()
        if not np.isnan(beta)
        else np.nan
    )

    tracking_error = (
        excess_return.std(ddof=1)
        * np.sqrt(annualization_factor)
    )

    information_ratio = (
        excess_return.mean()
        * annualization_factor
        / tracking_error
        if tracking_error > 0
        else np.nan
    )

    positive_benchmark = benchmark_return > 0
    negative_benchmark = benchmark_return < 0

    upside_capture = (
        strategy_return.loc[positive_benchmark].mean()
        / benchmark_return.loc[positive_benchmark].mean()
        * 100
        if positive_benchmark.any()
        else np.nan
    )

    downside_capture = (
        strategy_return.loc[negative_benchmark].mean()
        / benchmark_return.loc[negative_benchmark].mean()
        * 100
        if negative_benchmark.any()
        else np.nan
    )

    strategy_total = (
        data["Strategy_Equity"].iloc[-1]
        / data["Strategy_Equity"].iloc[0]
        - 1
    )
    benchmark_total = (
        data["Benchmark_Equity"].iloc[-1]
        / data["Benchmark_Equity"].iloc[0]
        - 1
    )

    return {
        "Alpha_Annual_%": (
            alpha_daily * annualization_factor * 100
            if not np.isnan(alpha_daily)
            else np.nan
        ),
        "Beta": beta,
        "Correlation": strategy_return.corr(
            benchmark_return
        ),
        "Tracking_Error_%": tracking_error * 100,
        "Information_Ratio": information_ratio,
        "Upside_Capture_%": upside_capture,
        "Downside_Capture_%": downside_capture,
        "Active_Total_Return_pp": (
            strategy_total - benchmark_total
        ) * 100,
        "Strategy_Annual_Volatility_%": (
            strategy_return.std(ddof=1)
            * np.sqrt(annualization_factor)
            * 100
        ),
        "Benchmark_Annual_Volatility_%": (
            benchmark_return.std(ddof=1)
            * np.sqrt(annualization_factor)
            * 100
        ),
    }


def drawdown_series(
    equity_df: pd.DataFrame,
    equity_column: str = "Equity",
) -> pd.DataFrame:
    """Return dated percentage drawdown observations."""
    required = {"Date", equity_column}
    missing = required.difference(equity_df.columns)

    if missing:
        raise KeyError(
            f"Drawdown için eksik sütunlar: {sorted(missing)}"
        )

    result = equity_df[
        ["Date", equity_column]
    ].sort_values("Date").copy()

    result["Peak"] = result[equity_column].cummax()
    result["Drawdown_%"] = (
        result[equity_column] / result["Peak"] - 1
    ) * 100

    return result[["Date", "Drawdown_%"]]


def calendar_return_table(
    curves: Mapping[str, pd.DataFrame],
    frequency: str = "YE",
) -> pd.DataFrame:
    """Create calendar returns from named equity curves."""
    records: list[pd.DataFrame] = []

    for name, equity_df in curves.items():
        required = {"Date", "Equity"}
        missing = required.difference(equity_df.columns)

        if missing:
            raise KeyError(
                f"{name} eğrisi için eksik sütunlar: "
                f"{sorted(missing)}"
            )

        curve = equity_df[
            ["Date", "Equity"]
        ].sort_values("Date").copy()

        curve["Date"] = pd.to_datetime(curve["Date"])
        curve = curve.set_index("Date")

        period_end = curve["Equity"].resample(
            frequency
        ).last().dropna()

        returns = period_end.pct_change(fill_method=None)

        if not period_end.empty:
            first_period_start = float(
                curve["Equity"].iloc[0]
            )
            returns.iloc[0] = (
                period_end.iloc[0]
                / first_period_start
                - 1
            )

        frame = returns.rename("Return").reset_index()
        frame["Portfolio"] = name
        records.append(frame)

    if not records:
        return pd.DataFrame(
            columns=["Date", "Return", "Portfolio"]
        )

    result = pd.concat(records, ignore_index=True)
    result["Return_%"] = result["Return"] * 100

    return result
