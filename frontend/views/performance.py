"""Concise promoted-strategy performance view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import APIClient
from frontend.formatters import format_percent, format_tl
from frontend.ui_helpers import render_line_chart, show_error, to_frame


def render_performance(client: APIClient) -> None:
    st.title("Performans")
    st.caption("Ana stratejinin BIST100 karşısındaki kilitli, tam geçmiş sonucu.")

    try:
        summary = client.get("/api/backtest/summary")
        periods = to_frame(summary.get("periods"))
        full = periods.loc[periods["Period"].eq("Full")]
        strategy = full.loc[full["Portfolio"].eq("Final Strategy")].iloc[0]
        benchmark = full.loc[full["Portfolio"].eq("BIST100 Gross")].iloc[0]

        metrics = st.columns(4)
        metrics[0].metric(
            "Strateji son değeri",
            format_tl(strategy["End_Value_TL"]),
        )
        metrics[1].metric(
            "CAGR",
            format_percent(strategy["CAGR_%"]),
            f"{strategy['CAGR_%'] - benchmark['CAGR_%']:+.2f} puan vs BIST100",
        )
        metrics[2].metric(
            "Maksimum düşüş",
            format_percent(strategy["Max_Drawdown_%"]),
        )
        metrics[3].metric(
            "BIST100 son değeri",
            format_tl(benchmark["End_Value_TL"]),
        )

        equity = to_frame(client.get("/api/backtest/equity"))
        equity["Date"] = pd.to_datetime(equity["Date"])
        render_line_chart(
            equity.set_index("Date"),
            ["Final_Strategy", "BIST100_Gross"],
            title="Baseline Robot ve BIST100",
        )

        with st.expander("Dönemsel ve yıllık ayrıntılar"):
            st.markdown("##### Dönemsel performans")
            st.dataframe(periods, width="stretch", hide_index=True)
            active = to_frame(summary.get("active_metrics"))
            if not active.empty:
                st.markdown("##### Aktif performans")
                st.dataframe(active, width="stretch", hide_index=True)
            yearly = to_frame(client.get("/api/backtest/yearly"))
            if not yearly.empty:
                st.markdown("##### Takvim yılı getirileri")
                st.dataframe(yearly, width="stretch", hide_index=True)
    except Exception as error:
        show_error(error)
        st.caption("Promote edilmiş backtest çıktılarının mevcut olduğundan emin olun.")
