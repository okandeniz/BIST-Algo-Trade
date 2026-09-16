"""Advanced comparisons kept away from the daily workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import APIClient
from frontend.ui_helpers import render_line_chart, show_error, to_frame


def render_research(client: APIClient) -> None:
    st.title("Araştırma")
    st.caption(
        "Challenger karşılaştırmaları ve model teşhisleri. Bu bölüm günlük işlem akışının parçası değildir."
    )

    comparison_tab, diagnostics_tab = st.tabs(
        ["Strateji karşılaştırması", "Model ve plan teşhisi"]
    )

    with comparison_tab:
        st.subheader("Ortak Audit dönemi")
        st.info(
            "RS126 ve ML sonuçları yalnızca ortak 2025+ Audit döneminde Baseline ile karşılaştırılır."
        )
        try:
            walk_forward = client.get("/api/backtest/walk-forward/summary")
            enhanced = client.get("/api/backtest/enhanced/summary")
            metrics = pd.concat(
                [
                    to_frame(walk_forward.get("metrics")),
                    to_frame(enhanced.get("metrics")),
                ],
                ignore_index=True,
                sort=False,
            )
            if not metrics.empty:
                st.dataframe(metrics, width="stretch", hide_index=True)

            wf_equity = to_frame(client.get("/api/backtest/walk-forward/equity"))
            enhanced_equity = to_frame(client.get("/api/backtest/enhanced/equity"))
            wf_equity["Date"] = pd.to_datetime(wf_equity["Date"])
            enhanced_equity["Date"] = pd.to_datetime(enhanced_equity["Date"])
            combined = wf_equity.merge(enhanced_equity, on="Date", how="inner")
            columns = [
                column
                for column in (
                    "Baseline_Robot",
                    "RS126_Enhanced",
                    "ML_Challenger",
                    "BIST100_Gross",
                )
                if column in combined
            ]
            render_line_chart(
                combined.set_index("Date"),
                columns,
                title="Ortak dönem strateji karşılaştırması",
            )

            with st.expander("Walk-forward ayrıntıları"):
                metadata = walk_forward.get("metadata", {})
                st.json(metadata)
                active = to_frame(walk_forward.get("active_metrics"))
                if not active.empty:
                    st.dataframe(active, width="stretch", hide_index=True)
                training = to_frame(
                    client.get("/api/backtest/walk-forward/training-log")
                )
                if not training.empty:
                    st.dataframe(training, width="stretch", hide_index=True)
        except Exception as error:
            show_error(error)
            st.caption("Araştırma artifact'leri üretilmemiş veya erişilemiyor olabilir.")

    with diagnostics_tab:
        try:
            latest = client.get("/api/signals/latest")
            metadata = latest.get("metadata", {})
            st.markdown("#### Son plan metadata")
            st.json(metadata)
            diagnostic = to_frame(latest.get("model_diagnostic"))
            if diagnostic.empty:
                st.caption("Model teşhisi bulunmuyor.")
            else:
                st.markdown("#### ML skor teşhisi")
                st.dataframe(diagnostic, width="stretch", hide_index=True)
            with st.expander("Ham alış/satış karşılaştırmaları"):
                buy = to_frame(latest.get("buy_comparison"))
                sell = to_frame(latest.get("sell_comparison"))
                if not buy.empty:
                    st.markdown("##### Alış")
                    st.dataframe(buy, width="stretch", hide_index=True)
                if not sell.empty:
                    st.markdown("##### Satış")
                    st.dataframe(sell, width="stretch", hide_index=True)
        except Exception as error:
            show_error(error)
