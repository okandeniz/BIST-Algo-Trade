"""Portfolio and benchmark performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> dict[str, float | int]:
    if equity_df.empty:
        raise ValueError("Equity tablosu boş.")

    equity = equity_df.sort_values("Date").copy()
    years = (
        (equity["Date"].max() - equity["Date"].min()).days / 365.25
    )
    start_value = float(equity["Equity"].iloc[0])
    end_value = float(equity["Equity"].iloc[-1])

    cagr = (
        (end_value / start_value) ** (1 / years) - 1
        if years > 0 and start_value > 0
        else np.nan
    )

    equity["Peak"] = equity["Equity"].cummax()
    equity["Drawdown"] = equity["Equity"] / equity["Peak"] - 1
    max_drawdown = float(equity["Drawdown"].min())

    daily_returns = equity["Equity"].pct_change(fill_method=None).dropna()
    daily_std = daily_returns.std(ddof=1)

    sharpe = (
        daily_returns.mean() / daily_std * np.sqrt(252)
        if daily_std and not np.isnan(daily_std)
        else np.nan
    )

    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std(ddof=1)
    sortino = (
        daily_returns.mean() / downside_std * np.sqrt(252)
        if downside_std and not np.isnan(downside_std)
        else np.nan
    )

    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    if trades_df.empty:
        gross_profit = gross_loss = 0.0
        profit_factor = np.nan
        win_rate = np.nan
        trade_count = 0
        expectancy = np.nan
        outlier_count = 0
    else:
        gross_profit = float(
            trades_df.loc[trades_df["Return"] > 0, "Return"].sum()
        )
        gross_loss = abs(float(
            trades_df.loc[trades_df["Return"] < 0, "Return"].sum()
        ))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else np.nan
        )
        win_rate = float((trades_df["Return"] > 0).mean())
        trade_count = int(len(trades_df))
        expectancy = float(trades_df["Return"].mean())
        outlier_count = int(
            trades_df.get("Is_Outlier", pd.Series(False, index=trades_df.index))
            .sum()
        )

    return {
        "Start_Value": start_value,
        "End_Value": end_value,
        "Total_Return_%": (end_value / start_value - 1) * 100,
        "CAGR_%": cagr * 100,
        "Max_Drawdown_%": max_drawdown * 100,
        "Profit_Factor": profit_factor,
        "Win_Rate_%": win_rate * 100 if not np.isnan(win_rate) else np.nan,
        "Expectancy_%": expectancy * 100 if not np.isnan(expectancy) else np.nan,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Trade_Count": trade_count,
        "Outlier_Count": outlier_count,
        "Exposure_%": float(
            equity["Open_Positions"].gt(0).mean() * 100
        ),
        "Average_Open_Positions": float(
            equity["Open_Positions"].mean()
        ),
        "Max_Open_Positions": int(
            equity["Open_Positions"].max()
        ),
        "Average_Invested_%": float(
            (
                (equity["Equity"] - equity["Cash"])
                / equity["Equity"]
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .mean()
            * 100
        ),
    }


def yearly_performance(equity_df: pd.DataFrame) -> pd.DataFrame:
    result = equity_df.sort_values("Date").copy()
    result["Year"] = result["Date"].dt.year
    records = []

    for year, group in result.groupby("Year"):
        group = group.copy()
        group["Peak"] = group["Equity"].cummax()
        drawdown = group["Equity"] / group["Peak"] - 1

        records.append({
            "Year": year,
            "Start_Equity": group["Equity"].iloc[0],
            "End_Equity": group["Equity"].iloc[-1],
            "Return_%": (
                group["Equity"].iloc[-1] / group["Equity"].iloc[0] - 1
            ) * 100,
            "Yearly_Max_DD_%": drawdown.min() * 100,
        })

    return pd.DataFrame(records)


def monthly_performance(equity_df: pd.DataFrame) -> pd.DataFrame:
    result = equity_df.sort_values("Date").copy()
    result["Month"] = result["Date"].dt.to_period("M")
    monthly = result.groupby("Month")["Equity"].agg(["first", "last"])
    monthly["Return_%"] = (monthly["last"] / monthly["first"] - 1) * 100
    return monthly.reset_index()
