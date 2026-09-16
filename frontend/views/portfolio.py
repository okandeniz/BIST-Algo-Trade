"""Focused portfolio tracking and transaction entry."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd
import streamlit as st

from frontend.api_client import APIClient
from frontend.formatters import format_tl
from src.product_config import (
    PRIMARY_STRATEGY,
    STRATEGIES,
    strategy_label,
    strategy_options,
)
from frontend.ui_helpers import (
    position_advice,
    render_portfolio_metrics,
    show_error,
    to_frame,
)


def _initialize_portfolio(
    client: APIClient,
    strategy_key: str,
    clear_cache: Callable[[], Any],
    default_capital: float,
) -> None:
    st.subheader("Portföyü başlat")
    st.info("İlk işlemden önce bu portföy için kullanılacak başlangıç sermayesini girin.")
    with st.form(f"initialize_{strategy_key}"):
        capital = st.number_input(
            "Başlangıç sermayesi (TL)",
            min_value=1.0,
            value=default_capital,
            step=10_000.0,
        )
        submitted = st.form_submit_button(
            "Portföyü başlat",
            type="primary",
            width="stretch",
        )
    if submitted:
        try:
            result = client.post(
                f"/api/portfolio/{strategy_key}/initial-capital",
                {"initial_capital": float(capital)},
            )
            clear_cache()
            st.success(result["message"])
            st.rerun()
        except Exception as error:
            show_error(error)


def _render_buy_form(
    client: APIClient,
    strategy_key: str,
    summary: dict[str, Any],
    planned_buys: pd.DataFrame,
    clear_cache: Callable[[], Any],
    commission_rate: float,
) -> None:
    options = ["Manuel giriş"]
    if not planned_buys.empty and "Ticker" in planned_buys:
        options.extend(planned_buys["Ticker"].dropna().astype(str).tolist())

    selected = st.selectbox(
        "Günlük plandan seç",
        options,
        key=f"buy_plan_{strategy_key}",
    )
    defaults: dict[str, Any] = {
        "ticker": "",
        "quantity": 1,
        "price": 1.0,
        "stop": 0.9,
        "score": None,
    }
    if selected != "Manuel giriş":
        row = planned_buys.loc[planned_buys["Ticker"].eq(selected)].iloc[0]
        defaults = {
            "ticker": selected,
            "quantity": max(1, int(row.get("Estimated_Shares", 1))),
            "price": float(row.get("Estimated_Entry", 1.0)),
            "stop": float(row.get("Estimated_Stop", 0.9)),
            "score": int(row.get("Score", 0)) or None,
        }

    with st.form(f"buy_{strategy_key}_{selected}"):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Hisse kodu", value=defaults["ticker"])
        quantity = c2.number_input(
            "Lot", min_value=1, value=defaults["quantity"], step=1
        )
        price = c3.number_input(
            "Alış fiyatı (TL)",
            min_value=0.01,
            value=defaults["price"],
            step=0.01,
            format="%.4f",
        )
        c4, c5 = st.columns(2)
        stop = c4.number_input(
            "Stop seviyesi (TL)",
            min_value=0.01,
            value=defaults["stop"],
            step=0.01,
            format="%.4f",
        )
        trade_date = c5.date_input("Alış tarihi", value=date.today())
        note = st.text_input("Not", value="")

        gross = int(quantity) * float(price)
        fee = round(gross * commission_rate, 2)
        remaining = float(summary.get("Cash_TL", 0)) - gross - fee
        st.caption(
            f"İşlem: {format_tl(gross)} · Komisyon: {format_tl(fee)} · "
            f"Tahmini kalan nakit: {format_tl(remaining)}"
        )
        submitted = st.form_submit_button(
            "Alışı kaydet",
            type="primary",
            width="stretch",
            disabled=remaining < 0,
        )

    if submitted:
        try:
            result = client.post(
                "/api/portfolio/buy",
                {
                    "portfolio_name": strategy_key,
                    "ticker": ticker,
                    "quantity": int(quantity),
                    "price": float(price),
                    "trade_date": trade_date.isoformat(),
                    "fees": fee,
                    "stop_loss": float(stop),
                    "signal_score": defaults["score"],
                    "note": note or None,
                },
            )
            clear_cache()
            st.success(
                f"{result['ticker']} alışı kaydedildi. Kalan nakit: "
                f"{format_tl(result['remaining_cash'])}"
            )
            st.rerun()
        except Exception as error:
            show_error(error)


def _render_sell_form(
    client: APIClient,
    strategy_key: str,
    positions: pd.DataFrame,
    clear_cache: Callable[[], Any],
    commission_rate: float,
) -> None:
    if positions.empty:
        st.info("Satılabilecek açık pozisyon yok.")
        return

    ticker = st.selectbox(
        "Satılacak hisse",
        positions["Ticker"].astype(str).tolist(),
        key=f"sell_ticker_{strategy_key}",
    )
    row = positions.loc[positions["Ticker"].eq(ticker)].iloc[0]
    maximum = int(row["Quantity"])
    latest_price = float(row["Latest_Price_TL"])

    with st.form(f"sell_{strategy_key}_{ticker}"):
        c1, c2, c3 = st.columns(3)
        quantity = c1.number_input(
            "Satış lotu",
            min_value=1,
            max_value=maximum,
            value=maximum,
            step=1,
        )
        price = c2.number_input(
            "Satış fiyatı (TL)",
            min_value=0.01,
            value=latest_price,
            step=0.01,
            format="%.4f",
        )
        trade_date = c3.date_input("Satış tarihi", value=date.today())
        note = st.text_input("Satış notu", value="")
        gross = int(quantity) * float(price)
        fee = round(gross * commission_rate, 2)
        st.caption(
            f"Satış: {format_tl(gross)} · Komisyon: {format_tl(fee)} · "
            f"Net nakit girişi: {format_tl(gross - fee)}"
        )
        submitted = st.form_submit_button(
            "Satışı kaydet",
            type="primary",
            width="stretch",
        )

    if submitted:
        try:
            result = client.post(
                "/api/portfolio/sell",
                {
                    "portfolio_name": strategy_key,
                    "ticker": ticker,
                    "quantity": int(quantity),
                    "price": float(price),
                    "trade_date": trade_date.isoformat(),
                    "fees": fee,
                    "note": note or None,
                },
            )
            clear_cache()
            st.success(
                f"{result['ticker']} satışı kaydedildi. Gerçekleşen K/Z: "
                f"{format_tl(result['realized_pnl'])}"
            )
            st.rerun()
        except Exception as error:
            show_error(error)


def render_portfolio(
    client: APIClient,
    clear_cache: Callable[[], Any],
    trading_config: dict[str, Any],
) -> None:
    st.title("Portföy")
    st.caption("Bir portföy seçin; pozisyonlar ve işlem formları yalnızca o portföy için gösterilir.")

    strategy_key = st.selectbox(
        "İzlenen strateji",
        strategy_options(),
        index=strategy_options().index(PRIMARY_STRATEGY),
        format_func=strategy_label,
    )
    strategy = STRATEGIES[strategy_key]
    portfolio_config = trading_config.get("portfolio", {})
    default_capital = float(
        portfolio_config.get("initial_capital", 500_000.0)
    )
    commission_rate = float(
        portfolio_config.get("commission_rate", 0.002)
    )
    if strategy_key != PRIMARY_STRATEGY:
        st.warning(f"{strategy.label} deneysel bir challenger portföyüdür.")

    try:
        summary = client.get(f"/api/portfolio/{strategy_key}/summary")
        positions = to_frame(client.get(f"/api/portfolio/{strategy_key}/positions"))
        transactions = to_frame(
            client.get(f"/api/portfolio/{strategy_key}/transactions?limit=500")
        )
        try:
            signals = client.get("/api/signals/latest")
        except Exception:
            signals = {}
    except Exception as error:
        show_error(error)
        return

    if not summary.get("Is_Initialized", False):
        _initialize_portfolio(
            client,
            strategy_key,
            clear_cache,
            default_capital,
        )
        return

    render_portfolio_metrics(summary)
    st.divider()

    heading, refresh_column = st.columns([5, 1.4])
    heading.subheader("Açık pozisyonlar")
    if refresh_column.button(
        "Fiyatları güncelle",
        width="stretch",
        disabled=positions.empty,
    ):
        try:
            with st.spinner("Son fiyatlar alınıyor…"):
                result = client.post(
                    f"/api/portfolio/{strategy_key}/refresh-prices",
                    timeout=180,
                )
            clear_cache()
            st.success(result["message"])
            st.rerun()
        except Exception as error:
            show_error(error)

    advised = position_advice(positions, signals, strategy.sell_plan_key)
    if advised.empty:
        st.info("Açık pozisyon yok.")
    else:
        st.dataframe(advised, width="stretch", hide_index=True)

    planned_buys = to_frame(signals.get(strategy.buy_plan_key))
    buy_tab, sell_tab = st.tabs(["Alış kaydet", "Satış kaydet"])
    with buy_tab:
        _render_buy_form(
            client,
            strategy_key,
            summary,
            planned_buys,
            clear_cache,
            commission_rate,
        )
    with sell_tab:
        _render_sell_form(
            client,
            strategy_key,
            positions,
            clear_cache,
            commission_rate,
        )

    with st.expander("İşlem geçmişi"):
        if transactions.empty:
            st.caption("Henüz işlem kaydı yok.")
        else:
            st.dataframe(transactions, width="stretch", hide_index=True)

    with st.expander("Gelişmiş: portföyü sıfırla"):
        st.warning("Bu işlem açık pozisyonları ve tüm işlem geçmişini kalıcı olarak siler.")
        confirmation = st.checkbox(
            "Silme işlemini onaylıyorum",
            key=f"reset_confirmation_{strategy_key}",
        )
        if st.button(
            "Portföyü sıfırla",
            disabled=not confirmation,
            key=f"reset_{strategy_key}",
        ):
            try:
                result = client.post(
                    f"/api/portfolio/{strategy_key}/reset",
                    {"confirmation": "RESET"},
                )
                clear_cache()
                st.success(result["message"])
                st.rerun()
            except Exception as error:
                show_error(error)
