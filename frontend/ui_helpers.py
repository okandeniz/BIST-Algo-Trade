"""Small presentation helpers shared by the Streamlit views."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.formatters import format_percent, format_tl


PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
}


def to_frame(data: Any) -> pd.DataFrame:
    return pd.DataFrame(data or [])


def format_date(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "—" if value is None else str(value)
    return timestamp.strftime("%d.%m.%Y")


def show_error(error: Exception) -> None:
    st.error(str(error))


def render_portfolio_metrics(summary: dict[str, Any]) -> None:
    if not summary.get("Is_Initialized", False):
        st.info("Portföy henüz başlatılmadı.")
        return

    columns = st.columns(5)
    columns[0].metric(
        "Portföy değeri",
        format_tl(summary.get("Equity_TL", 0)),
        format_percent(summary.get("Total_Return_%", 0)),
    )
    columns[1].metric("Nakit", format_tl(summary.get("Cash_TL", 0)))
    columns[2].metric(
        "Gerçekleşmemiş K/Z",
        format_tl(summary.get("Unrealized_PnL_TL", 0)),
    )
    columns[3].metric(
        "Gerçekleşen K/Z",
        format_tl(summary.get("Realized_PnL_TL", 0)),
    )
    columns[4].metric(
        "Açık pozisyon",
        int(summary.get("Open_Positions", 0)),
    )


def render_line_chart(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    title: str,
    height: int = 460,
) -> None:
    figure = go.Figure()
    for column in columns:
        if column not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                name=column.replace("_", " "),
                mode="lines",
            )
        )
    figure.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=True),
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config=PLOTLY_CONFIG,
    )


def position_advice(
    positions: pd.DataFrame,
    signals: dict[str, Any],
    sell_plan_key: str,
) -> pd.DataFrame:
    if positions.empty:
        return positions

    sell_orders = to_frame(signals.get(sell_plan_key))
    sell_reasons: dict[str, str] = {}
    if not sell_orders.empty and "Ticker" in sell_orders:
        sell_reasons = {
            str(row["Ticker"]): str(row.get("Reason", "Strateji çıkışı"))
            for _, row in sell_orders.iterrows()
        }

    result = positions.copy()
    advice: list[str] = []
    for _, row in result.iterrows():
        ticker = str(row["Ticker"])
        price = float(row["Latest_Price_TL"])
        stop = float(row["Stop_Loss_TL"])
        if price <= stop:
            advice.append("🔴 Stop altında — satışı değerlendir")
        elif ticker in sell_reasons:
            advice.append(f"🟠 Çıkış sinyali — {sell_reasons[ticker]}")
        else:
            advice.append("🟢 Tut")
    result.insert(1, "Durum", advice)
    return result
