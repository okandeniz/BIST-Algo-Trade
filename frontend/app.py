"""Streamlit dashboard for the BIST100 Robot project."""

from __future__ import annotations

from datetime import date
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import APIClient
from formatters import (
    format_percent,
    format_tl,
)


st.set_page_config(
    page_title="BIST100 Robot",
    page_icon="📈",
    layout="wide",
)

st.title("BIST100 Robot — Backtest, Sinyal ve Portföy Takibi")

default_api_url = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

api_url = st.sidebar.text_input(
    "FastAPI adresi",
    value=default_api_url,
)

client = APIClient(api_url)


def to_frame(data: Any) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)



def show_error(error: Exception) -> None:
    st.error(str(error))


PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": False,
}


def render_line_chart(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    title: str,
    height: int = 500,
) -> None:
    """Render a horizontal-pan-only time-series chart."""
    figure = go.Figure()

    for column in columns:
        if column not in frame.columns:
            continue

        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=column.replace("_", " "),
            )
        )

    figure.update_layout(
        title=title,
        height=height,
        dragmode="pan",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )
    figure.update_xaxes(
        fixedrange=False,
        showspikes=True,
        spikemode="across",
    )
    figure.update_yaxes(
        fixedrange=True,
        tickformat=",",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def render_bar_chart(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    title: str,
    height: int = 420,
) -> None:
    """Render bars without zoom controls and with horizontal panning."""
    figure = go.Figure()

    for column in columns:
        if column not in frame.columns:
            continue

        figure.add_trace(
            go.Bar(
                x=frame.index,
                y=frame[column],
                name=column.replace("_", " "),
            )
        )

    figure.update_layout(
        title=title,
        height=height,
        barmode="group",
        dragmode="pan",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )
    figure.update_xaxes(fixedrange=False)
    figure.update_yaxes(fixedrange=True)

    st.plotly_chart(
        figure,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def portfolio_label(portfolio_name: str) -> str:
    return portfolio_name.replace("_", " ")


def format_metric_delta_percent(
    value: float | int | None,
) -> str:
    """Format metric delta with the sign first for Streamlit coloring."""
    if value is None:
        return "0,00%"

    numeric_value = float(value)

    if abs(numeric_value) < 0.005:
        return "0,00%"

    formatted = f"{numeric_value:+.2f}".replace(".", ",")
    return f"{formatted}%"


def render_summary_metrics(summary: dict[str, Any]) -> None:
    if not summary.get("Is_Initialized", False):
        st.info(
            "Bu portföy henüz başlatılmadı. İşlem açmadan önce "
            "kullanılacak başlangıç sermayesini girin."
        )
        return

    first, second, third, fourth, fifth = st.columns(5)

    first.metric(
        "Portföy değeri",
        format_tl(summary["Equity_TL"]),
        delta=format_metric_delta_percent(
            summary["Total_Return_%"]
        ),
        delta_color="normal",
    )
    second.metric(
        "Nakit",
        format_tl(summary["Cash_TL"]),
    )
    third.metric(
        "Gerçekleşen K/Z",
        format_tl(summary["Realized_PnL_TL"]),
    )
    fourth.metric(
        "Gerçekleşmemiş K/Z",
        format_tl(summary["Unrealized_PnL_TL"]),
    )
    fifth.metric(
        "Açık pozisyon",
        int(summary["Open_Positions"]),
    )

    st.caption(
        f"Başlangıç nakdi: "
        f"{format_tl(summary['Initial_Capital_TL'])} · "
        f"Yatırım oranı: "
        f"{format_percent(summary['Invested_%'])} · "
        f"İşlem kaydı: "
        f"{int(summary.get('Transaction_Count', 0))}"
    )


def add_position_advice(
    positions: pd.DataFrame,
    latest_signals: dict[str, Any],
    portfolio_name: str,
) -> pd.DataFrame:
    """Attach stop and strategy-exit recommendations to open positions."""
    if positions.empty:
        return positions

    sell_key = (
        "baseline_sells"
        if portfolio_name == "Baseline_Robot"
        else "challenger_sells"
    )
    sell_orders = to_frame(
        latest_signals.get(sell_key)
    )

    sell_reasons: dict[str, str] = {}

    if (
        not sell_orders.empty
        and "Ticker" in sell_orders.columns
    ):
        for _, row in sell_orders.iterrows():
            reason = str(
                row.get("Reason", "Strateji çıkış sinyali")
            )
            sell_reasons[str(row["Ticker"])] = reason

    result = positions.copy()
    recommendations: list[str] = []

    for _, row in result.iterrows():
        ticker = str(row["Ticker"])
        latest_price = float(row["Latest_Price_TL"])
        stop_loss = float(row["Stop_Loss_TL"])
        unrealized_pnl = float(
            row["Unrealized_PnL_TL"]
        )

        if latest_price <= stop_loss:
            recommendation = (
                "🔴 SATIŞ TAVSİYESİ — Stop loss altında"
            )
        elif ticker in sell_reasons:
            if unrealized_pnl > 0:
                recommendation = (
                    "🟠 KÂR AL / SATIŞ TAVSİYESİ — "
                    + sell_reasons[ticker]
                )
            else:
                recommendation = (
                    "🟠 SATIŞ TAVSİYESİ — "
                    + sell_reasons[ticker]
                )
        else:
            recommendation = "🟢 TUT"

        recommendations.append(recommendation)

    result["Tavsiye"] = recommendations

    preferred_order = [
        "Ticker",
        "Quantity",
        "Average_Cost_TL",
        "Latest_Price_TL",
        "Stop_Loss_TL",
        "Distance_To_Stop_%",
        "Unrealized_PnL_TL",
        "Unrealized_PnL_%",
        "Market_Value_TL",
        "Entry_Date",
        "Signal_Score",
        "Tavsiye",
    ]

    return result[
        [
            column
            for column in preferred_order
            if column in result.columns
        ]
    ]


def show_position_alerts(
    advised_positions: pd.DataFrame,
) -> None:
    if advised_positions.empty:
        return

    stop_rows = advised_positions.loc[
        advised_positions["Tavsiye"].str.contains(
            "Stop loss",
            na=False,
        )
    ]
    take_profit_rows = advised_positions.loc[
        advised_positions["Tavsiye"].str.contains(
            "KÂR AL",
            na=False,
        )
    ]

    if not stop_rows.empty:
        tickers = ", ".join(
            stop_rows["Ticker"].astype(str)
        )
        st.error(
            f"Stop loss altına inen pozisyonlar: {tickers}. "
            "Satış işlemi değerlendirilmelidir."
        )

    if not take_profit_rows.empty:
        tickers = ", ".join(
            take_profit_rows["Ticker"].astype(str)
        )
        st.warning(
            f"Kâr alma / strateji çıkış sinyali bulunan "
            f"pozisyonlar: {tickers}."
        )


def render_portfolio_workspace(
    portfolio_name: str,
    latest_signals: dict[str, Any],
) -> None:
    """Render one fully independent paper-trading portfolio."""
    try:
        summary = client.get(
            f"/api/portfolio/{portfolio_name}/summary"
        )
        positions = to_frame(
            client.get(
                f"/api/portfolio/{portfolio_name}/positions"
            )
        )
        transactions = to_frame(
            client.get(
                f"/api/portfolio/{portfolio_name}/transactions"
            )
        )
    except Exception as error:
        show_error(error)
        return

    if not summary.get("Is_Initialized", False):
        st.markdown("### Portföyü başlat")
        st.info(
            "Alış işlemi açılmadan önce bu portföyde kullanılacak "
            "gerçek başlangıç sermayesini girin. Sistem artık "
            "500.000 TL varsayımı kullanmaz."
        )

        with st.form(
            f"capital_setup_form_{portfolio_name}"
        ):
            capital_value = st.number_input(
                "Kullanılacak başlangıç sermayesi (TL)",
                min_value=0.0,
                value=0.0,
                step=10_000.0,
                format="%.2f",
                key=f"capital_setup_input_{portfolio_name}",
                help=(
                    "Örneğin 300.000 TL ile başlayacaksanız "
                    "300000 girin."
                ),
            )
            capital_submit = st.form_submit_button(
                "Portföyü bu sermayeyle başlat",
                type="primary",
                use_container_width=True,
            )

        if capital_submit:
            if capital_value <= 0:
                st.error(
                    "Başlangıç sermayesi sıfırdan büyük olmalıdır."
                )
            else:
                try:
                    result = client.post(
                        f"/api/portfolio/"
                        f"{portfolio_name}/initial-capital",
                        {
                            "initial_capital": float(
                                capital_value
                            )
                        },
                    )
                    st.success(result["message"])
                    st.rerun()
                except Exception as error:
                    show_error(error)

        return

    render_summary_metrics(summary)

    with st.expander(
        "Portföyü sıfırlama",
        expanded=False,
    ):
        st.warning(
            "Sıfırlama açık pozisyonları ve bütün alış/satış "
            "kayıtlarını kalıcı olarak siler. Sonrasında yeni "
            "işlem açmadan önce başlangıç sermayesini tekrar "
            "girmeniz gerekir."
        )

        with st.form(
            f"reset_form_{portfolio_name}"
        ):
            reset_confirm = st.checkbox(
                "Bütün alış, satış ve açık pozisyon "
                "kayıtlarının silinmesini onaylıyorum.",
                key=f"reset_confirm_{portfolio_name}",
            )
            reset_submit = st.form_submit_button(
                "Portföyü sıfırla",
                use_container_width=True,
            )

        if reset_submit:
            if not reset_confirm:
                st.error(
                    "Sıfırlama için onay kutusunu işaretleyin."
                )
            else:
                try:
                    result = client.post(
                        f"/api/portfolio/"
                        f"{portfolio_name}/reset",
                        {
                            "confirmation": "RESET",
                        },
                    )
                    st.success(result["message"])
                    st.rerun()
                except Exception as error:
                    show_error(error)

    position_title, refresh_column = st.columns(
        [5, 1.45]
    )

    with position_title:
        st.markdown("#### Açık pozisyonlar")

    refresh_clicked = refresh_column.button(
        "🔄 Son fiyatları güncelle",
        key=f"refresh_prices_{portfolio_name}",
        use_container_width=True,
        disabled=positions.empty,
        help=(
            "Yalnızca açık pozisyonların Son fiyat alanını "
            "Yahoo Finance verisiyle günceller."
        ),
    )

    if refresh_clicked:
        try:
            with st.spinner(
                "Açık pozisyonların fiyatları alınıyor..."
            ):
                refresh_result = client.post(
                    f"/api/portfolio/"
                    f"{portfolio_name}/refresh-prices",
                    timeout=90.0,
                )

            st.session_state[
                f"price_refresh_result_{portfolio_name}"
            ] = refresh_result
            st.rerun()
        except Exception as error:
            show_error(error)

    refresh_result = st.session_state.pop(
        f"price_refresh_result_{portfolio_name}",
        None,
    )

    if refresh_result:
        if refresh_result.get(
            "updated_count",
            0,
        ) > 0:
            st.success(
                refresh_result.get(
                    "message",
                    "Son fiyatlar güncellendi.",
                )
            )

        if refresh_result.get(
            "failed_count",
            0,
        ) > 0:
            failed_tickers = [
                str(record.get("Ticker"))
                for record in refresh_result.get(
                    "quotes",
                    [],
                )
                if record.get("Status") != "OK"
            ]
            st.warning(
                "Fiyatı alınamayan hisseler: "
                + ", ".join(failed_tickers)
            )

        successful_quote_times = [
            str(record.get("Quote_Time"))
            for record in refresh_result.get(
                "quotes",
                [],
            )
            if (
                record.get("Status") == "OK"
                and record.get("Quote_Time")
            )
        ]

        if successful_quote_times:
            st.caption(
                "En son fiyat zamanı: "
                + max(successful_quote_times)
                + " · Kaynak: Yahoo Finance"
            )

    advised_positions = add_position_advice(
        positions,
        latest_signals,
        portfolio_name,
    )
    show_position_alerts(advised_positions)

    if advised_positions.empty:
        st.info("Henüz açık pozisyon bulunmuyor.")
    else:
        st.dataframe(
            advised_positions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Quantity": st.column_config.NumberColumn(
                    "Lot",
                    format="%d",
                ),
                "Average_Cost_TL": st.column_config.NumberColumn(
                    "Ortalama maliyet",
                    format="%.4f TL",
                ),
                "Latest_Price_TL": st.column_config.NumberColumn(
                    "Son fiyat",
                    format="%.4f TL",
                ),
                "Stop_Loss_TL": st.column_config.NumberColumn(
                    "Stop loss",
                    format="%.4f TL",
                ),
                "Distance_To_Stop_%": st.column_config.NumberColumn(
                    "Stop mesafesi",
                    format="%.2f%%",
                ),
                "Unrealized_PnL_TL": st.column_config.NumberColumn(
                    "Gerçekleşmemiş K/Z",
                    format="%.2f TL",
                ),
                "Unrealized_PnL_%": st.column_config.NumberColumn(
                    "K/Z %",
                    format="%.2f%%",
                ),
                "Market_Value_TL": st.column_config.NumberColumn(
                    "Piyasa değeri",
                    format="%.2f TL",
                ),
                "Tavsiye": st.column_config.TextColumn(
                    "Durum / Tavsiye",
                    width="large",
                ),
            },
        )
        st.caption(
            "Kâr al tavsiyesi, günlük Robot satış planı oluştuğunda "
            "ve pozisyon kârdayken gösterilir. Stop tavsiyesi, son "
            "fiyat kayıtlı stop seviyesinin altına indiğinde gösterilir. "
            "Fiyat güncelleme düğmesi yalnızca Son fiyat değerlerini "
            "kaydeder; Yahoo Finance verisi gecikmeli olabilir."
        )

    buy_tab, sell_tab = st.tabs(
        [
            "Alış kaydı",
            "Satış kaydı",
        ]
    )

    plan_key = (
        "baseline_buys"
        if portfolio_name == "Baseline_Robot"
        else "challenger_buys"
    )
    planned_buys = to_frame(
        latest_signals.get(plan_key)
    )

    with buy_tab:
        signal_options = ["Manuel giriş"]

        if (
            not planned_buys.empty
            and "Ticker" in planned_buys.columns
        ):
            signal_options.extend(
                planned_buys["Ticker"]
                .dropna()
                .astype(str)
                .tolist()
            )

        selected_signal = st.selectbox(
            "Günlük plandan seç",
            signal_options,
            key=f"buy_signal_{portfolio_name}",
        )

        default_ticker = ""
        default_quantity = 1
        default_price = 1.0
        default_stop = 0.9
        default_score = 0

        if (
            selected_signal != "Manuel giriş"
            and not planned_buys.empty
        ):
            selected_row = planned_buys.loc[
                planned_buys["Ticker"].eq(
                    selected_signal
                )
            ].iloc[0]

            default_ticker = selected_signal
            default_quantity = max(
                1,
                int(
                    selected_row.get(
                        "Estimated_Shares",
                        1,
                    )
                ),
            )
            default_price = float(
                selected_row.get(
                    "Estimated_Entry",
                    1.0,
                )
            )
            default_stop = float(
                selected_row.get(
                    "Estimated_Stop",
                    default_price * 0.90,
                )
            )
            default_score = int(
                selected_row.get("Score", 0)
            )

        form_key = (
            f"{portfolio_name}_{selected_signal}"
        )

        with st.form(
            f"buy_form_{form_key}",
            clear_on_submit=False,
        ):
            b1, b2, b3 = st.columns(3)

            ticker = b1.text_input(
                "Hisse kodu",
                value=default_ticker,
            )
            quantity = b2.number_input(
                "Lot",
                min_value=1,
                value=default_quantity,
                step=1,
            )
            price = b3.number_input(
                "Alış fiyatı (TL)",
                min_value=0.01,
                value=default_price,
                step=0.01,
                format="%.4f",
            )

            b4, b5, b6 = st.columns(3)
            stop_loss = b4.number_input(
                "Stop seviyesi (TL)",
                min_value=0.01,
                value=default_stop,
                step=0.01,
                format="%.4f",
            )
            gross_amount = (
                int(quantity) * float(price)
            )
            fees = round(
                gross_amount * 0.002,
                2,
            )

            b5.number_input(
                "Toplam masraf — binde 2 (TL)",
                min_value=0.0,
                value=fees,
                step=0.01,
                format="%.2f",
                disabled=True,
                help=(
                    "Komisyon otomatik hesaplanır: "
                    "Lot × alış fiyatı × 0,002"
                ),
            )
            trade_date = b6.date_input(
                "Alış tarihi",
                value=date.today(),
            )

            total_cash_out = (
                gross_amount + fees
            )
            estimated_remaining_cash = (
                float(summary["Cash_TL"])
                - total_cash_out
            )

            st.caption(
                f"İşlem tutarı: "
                f"{format_tl(gross_amount)} · "
                f"Komisyon: "
                f"{format_tl(fees)} · "
                f"Toplam nakit çıkışı: "
                f"{format_tl(total_cash_out)} · "
                f"Tahmini kalan nakit: "
                f"{format_tl(estimated_remaining_cash)}"
            )

            if estimated_remaining_cash < 0:
                st.error(
                    "Girilen lot ve fiyat mevcut nakdi aşıyor."
                )

            note = st.text_input(
                "Not",
                value=(
                    f"Günlük sinyal skoru: {default_score}"
                    if default_score
                    else ""
                ),
            )

            buy_submit = st.form_submit_button(
                "Alışı kaydet",
                type="primary",
                use_container_width=True,
                disabled=estimated_remaining_cash < 0,
            )

        if buy_submit:
            try:
                result = client.post(
                    "/api/portfolio/buy",
                    {
                        "portfolio_name": portfolio_name,
                        "ticker": ticker,
                        "quantity": int(quantity),
                        "price": float(price),
                        "trade_date": (
                            trade_date.isoformat()
                        ),
                        "fees": float(fees),
                        "stop_loss": float(stop_loss),
                        "signal_score": (
                            default_score
                            if default_score
                            else None
                        ),
                        "note": note or None,
                    },
                )
                st.success(
                    f"{result['ticker']} alış kaydı oluşturuldu. "
                    f"Kalan nakit: "
                    f"{format_tl(result['remaining_cash'])}"
                )
                st.rerun()
            except Exception as error:
                show_error(error)

    with sell_tab:
        if positions.empty:
            st.info(
                "Satılabilecek açık pozisyon bulunmuyor."
            )
        else:
            sell_ticker = st.selectbox(
                "Satılacak hisse",
                positions["Ticker"].tolist(),
                key=f"sell_ticker_{portfolio_name}",
            )

            selected_rows = positions.loc[
                positions["Ticker"].eq(
                    sell_ticker
                )
            ]

            if selected_rows.empty:
                st.error(
                    "Seçilen hisseye ait açık pozisyon bulunamadı."
                )
            else:
                position_row = selected_rows.iloc[0]
                max_quantity = int(
                    position_row["Quantity"]
                )
                latest_price = float(
                    position_row["Latest_Price_TL"]
                )

                quantity_key = (
                    f"sell_quantity_"
                    f"{portfolio_name}_{sell_ticker}"
                )
                price_key = (
                    f"sell_price_"
                    f"{portfolio_name}_{sell_ticker}"
                )
                date_key = (
                    f"sell_date_"
                    f"{portfolio_name}_{sell_ticker}"
                )
                note_key = (
                    f"sell_note_"
                    f"{portfolio_name}_{sell_ticker}"
                )

                # Her ticker farklı bir widget durumu kullanır.
                # Bir hissenin lotu diğer hisseye taşınmaz.
                saved_quantity = st.session_state.get(
                    quantity_key
                )

                if (
                    saved_quantity is None
                    or int(saved_quantity) < 1
                    or int(saved_quantity) > max_quantity
                ):
                    st.session_state[
                        quantity_key
                    ] = max_quantity

                if price_key not in st.session_state:
                    st.session_state[
                        price_key
                    ] = latest_price

                s1, s2, s3 = st.columns(3)

                sell_quantity = s1.number_input(
                    "Satış lotu",
                    min_value=1,
                    max_value=max_quantity,
                    step=1,
                    key=quantity_key,
                    help=(
                        "Varsayılan değer seçilen hissenin "
                        "açık pozisyondaki toplam lotudur. "
                        "Kısmi satış için azaltabilirsiniz."
                    ),
                )

                sell_price = s2.number_input(
                    "Satış fiyatı (TL)",
                    min_value=0.01,
                    step=0.01,
                    format="%.4f",
                    key=price_key,
                )

                gross_sale_amount = (
                    int(sell_quantity)
                    * float(sell_price)
                )
                sell_fees = round(
                    gross_sale_amount * 0.002,
                    2,
                )
                net_sale_proceeds = (
                    gross_sale_amount - sell_fees
                )

                s3.metric(
                    "Toplam masraf — binde 2",
                    format_tl(sell_fees),
                )

                st.caption(
                    f"Açık pozisyon: {max_quantity:,} lot · "
                    f"Satış tutarı: "
                    f"{format_tl(gross_sale_amount)} · "
                    f"Komisyon: {format_tl(sell_fees)} · "
                    f"Net nakit girişi: "
                    f"{format_tl(net_sale_proceeds)}"
                )

                s4, s5 = st.columns(2)

                sell_date = s4.date_input(
                    "Satış tarihi",
                    value=date.today(),
                    key=date_key,
                )

                sell_note = s5.text_input(
                    "Satış notu",
                    value="",
                    key=note_key,
                )

                sell_submit = st.button(
                    "Satışı kaydet",
                    type="primary",
                    use_container_width=True,
                    key=(
                        f"sell_submit_"
                        f"{portfolio_name}_{sell_ticker}"
                    ),
                )

                if sell_submit:
                    try:
                        result = client.post(
                            "/api/portfolio/sell",
                            {
                                "portfolio_name": (
                                    portfolio_name
                                ),
                                "ticker": sell_ticker,
                                "quantity": int(
                                    sell_quantity
                                ),
                                "price": float(
                                    sell_price
                                ),
                                "trade_date": (
                                    sell_date.isoformat()
                                ),
                                "fees": float(
                                    sell_fees
                                ),
                                "note": (
                                    sell_note or None
                                ),
                            },
                        )

                        st.success(
                            f"{result['ticker']} satışı "
                            f"kaydedildi. Komisyon: "
                            f"{format_tl(result['fees'])} · "
                            f"Gerçekleşen K/Z: "
                            f"{format_tl(result['realized_pnl'])}"
                        )

                        st.session_state.pop(
                            quantity_key,
                            None,
                        )
                        st.session_state.pop(
                            price_key,
                            None,
                        )
                        st.session_state.pop(
                            note_key,
                            None,
                        )
                        st.rerun()

                    except Exception as error:
                        show_error(error)

    st.markdown("#### İşlem geçmişi")
    if transactions.empty:
        st.info("Henüz alış veya satış kaydı bulunmuyor.")
    else:
        st.dataframe(
            transactions,
            use_container_width=True,
            hide_index=True,
        )

        realized = transactions.loc[
            transactions["side"].eq("SELL")
        ].copy()

        st.markdown("#### Gerçekleşen kâr-zarar")
        if realized.empty:
            st.info(
                "Henüz kapanmış işlem bulunmuyor."
            )
        else:
            st.dataframe(
                realized[
                    [
                        "trade_date",
                        "ticker",
                        "quantity",
                        "price",
                        "fees",
                        "realized_pnl",
                        "note",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


try:
    health = client.get("/health", timeout=5)
    st.sidebar.success("FastAPI bağlantısı aktif")
    st.sidebar.caption(
        f"Piyasa verisi: "
        f"{health['market_data'].get('latest_date') or 'Yok'}"
    )
except Exception as error:
    st.sidebar.error("FastAPI bağlantısı kurulamadı")
    st.sidebar.code(
        "uvicorn backend.app.main:app "
        "--reload --port 8000"
    )
    show_error(error)
    st.stop()


tab_overview, tab_backtest, tab_signals, tab_portfolio = st.tabs(
    [
        "Genel Bakış",
        "Backtest vs BIST100",
        "Günlük Sinyaller",
        "Alış / Satış ve Kâr-Zarar",
    ]
)



with tab_overview:
    st.subheader("Portföy özeti")

    try:
        latest_signals_overview = client.get(
            "/api/signals/latest"
        )
    except Exception:
        latest_signals_overview = {}

    overview_tabs = st.tabs(
        [
            "Baseline Robot",
            "ML Challenger",
        ]
    )

    for overview_tab, portfolio_name in zip(
        overview_tabs,
        [
            "Baseline_Robot",
            "ML_Challenger",
        ],
        strict=True,
    ):
        with overview_tab:
            st.markdown(
                f"### {portfolio_label(portfolio_name)}"
            )
            try:
                overview_summary = client.get(
                    f"/api/portfolio/"
                    f"{portfolio_name}/summary"
                )
                render_summary_metrics(
                    overview_summary
                )
            except Exception as error:
                show_error(error)

    st.divider()
    st.subheader("Son günlük sinyal özeti")

    if latest_signals_overview:
        metadata = latest_signals_overview.get(
            "metadata",
            {},
        )

        if metadata and not metadata.get(
            "trade_ready",
            True,
        ):
            st.warning(
                "Hisse ve model-ready tarih aynı değil. "
                "Sinyaller analiz amaçlı gösteriliyor; "
                "güncel emir olarak uygulanmamalı."
            )

        st.caption(
            f"Sinyal tarihi: "
            f"{latest_signals_overview.get('signal_date')} · "
            f"Son hisse tarihi: "
            f"{metadata.get('latest_stock_date', '—')}"
        )

        st.dataframe(
            to_frame(
                latest_signals_overview.get("summary")
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Henüz günlük sinyal planı oluşturulmamış."
        )


with tab_backtest:
    st.subheader(
        "Final strateji ve BIST100 performans karşılaştırması"
    )

    try:
        backtest = client.get(
            "/api/backtest/summary"
        )
        periods = to_frame(backtest["periods"])
        active = to_frame(
            backtest["active_metrics"]
        )

        full = periods.loc[
            periods["Period"].eq("Full")
        ].copy()

        strategy_row = full.loc[
            full["Portfolio"].eq("Final Strategy")
        ].iloc[0]
        bist_row = full.loc[
            full["Portfolio"].eq("BIST100 Gross")
        ].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Final strateji son değer",
            format_tl(strategy_row["End_Value_TL"]),
        )
        c2.metric(
            "BIST100 son değer",
            format_tl(bist_row["End_Value_TL"]),
        )
        c3.metric(
            "Strateji CAGR",
            format_percent(strategy_row["CAGR_%"]),
            (
                f"{strategy_row['CAGR_%'] - bist_row['CAGR_%']:.2f} "
                "puan"
            ),
        )
        c4.metric(
            "Strateji Max DD",
            format_percent(
                strategy_row["Max_Drawdown_%"]
            ),
            (
                f"{strategy_row['Max_Drawdown_%'] - bist_row['Max_Drawdown_%']:.2f} "
                "puan"
            ),
        )

        equity = to_frame(
            client.get("/api/backtest/equity")
        )
        equity["Date"] = pd.to_datetime(
            equity["Date"]
        )

        chart = equity.set_index("Date")[
            [
                "Final_Strategy",
                "BIST100_Gross",
                "BIST100_Net",
            ]
        ]
        render_line_chart(
            chart,
            [
                "Final_Strategy",
                "BIST100_Gross",
                "BIST100_Net",
            ],
            title=(
                "500.000 TL — Baseline Robot ve BIST100"
            ),
            height=500,
        )

        st.markdown("#### Dönemsel performans")
        st.dataframe(
            periods,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Aktif performans")
        st.dataframe(
            active,
            use_container_width=True,
            hide_index=True,
        )

        yearly = to_frame(
            client.get("/api/backtest/yearly")
        )
        if not yearly.empty:
            yearly = yearly.set_index("Year")
            available = [
                column
                for column in (
                    "Final Strategy",
                    "BIST100 Gross",
                )
                if column in yearly.columns
            ]
            if available:
                st.markdown(
                    "#### Takvim yılı getirileri (%)"
                )
                render_bar_chart(
                    yearly[available],
                    available,
                    title="Takvim yılı getirileri (%)",
                )


    except Exception as error:
        show_error(error)

    st.divider()
    st.subheader(
        "Walk-forward: Baseline Robot – ML Challenger – BIST100"
    )
    st.caption(
        "Karşılaştırma 2025'te başlar. Model her ay yalnızca geçmişte "
        "kapanmış işlemlerle yeniden eğitilir ve 5 günlük embargo "
        "uygulanır. Model tipi ile Q40 eşiği değiştirilmez."
    )

    try:
        wf_summary = client.get(
            "/api/backtest/walk-forward/summary"
        )
        wf_metadata = wf_summary.get(
            "metadata",
            {},
        )
        wf_metrics = to_frame(
            wf_summary.get("metrics")
        )
        wf_active = to_frame(
            wf_summary.get("active_metrics")
        )

        wf1, wf2, wf3, wf4 = st.columns(4)
        wf1.metric(
            "Dönem",
            (
                f"{wf_metadata.get('start', '—')} "
                f"→ {wf_metadata.get('end', '—')}"
            ),
        )
        wf2.metric(
            "Model",
            wf_metadata.get("model", "—"),
        )
        wf3.metric(
            "Hedef",
            wf_metadata.get("target", "—"),
        )
        wf4.metric(
            "Kilitli eşik",
            (
                f"{float(wf_metadata['probability_threshold']):.6f}"
                if wf_metadata.get(
                    "probability_threshold"
                ) is not None
                else "—"
            ),
        )

        st.warning(
            wf_metadata.get(
                "warning",
                "Bu dönem kusursuz bir holdout değildir.",
            )
        )

        if not wf_metrics.empty:
            metric_index = wf_metrics.set_index(
                "Portfolio"
            )

            baseline_wf = metric_index.loc[
                "Baseline_Robot"
            ]
            challenger_wf = metric_index.loc[
                "ML_Challenger"
            ]
            bist_wf = metric_index.loc[
                "BIST100_Gross"
            ]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Baseline CAGR",
                format_percent(
                    baseline_wf["CAGR_%"]
                ),
            )
            k2.metric(
                "ML CAGR",
                format_percent(
                    challenger_wf["CAGR_%"]
                ),
                (
                    f"{challenger_wf['CAGR_%'] - baseline_wf['CAGR_%']:.2f} "
                    "puan"
                ),
            )
            k3.metric(
                "ML Max DD",
                format_percent(
                    challenger_wf[
                        "Max_Drawdown_%"
                    ]
                ),
                (
                    f"{challenger_wf['Max_Drawdown_%'] - baseline_wf['Max_Drawdown_%']:.2f} "
                    "puan"
                ),
            )
            k4.metric(
                "BIST100 CAGR",
                format_percent(
                    bist_wf["CAGR_%"]
                ),
            )

        wf_equity = to_frame(
            client.get(
                "/api/backtest/walk-forward/equity"
            )
        )
        wf_equity["Date"] = pd.to_datetime(
            wf_equity["Date"]
        )

        walk_forward_chart = (
            wf_equity.set_index("Date")[
                [
                    "Baseline_Robot",
                    "ML_Challenger",
                    "BIST100_Gross",
                ]
            ]
        )
        render_line_chart(
            walk_forward_chart,
            [
                "Baseline_Robot",
                "ML_Challenger",
                "BIST100_Gross",
            ],
            title=(
                "Walk-forward — Baseline, ML Challenger "
                "ve BIST100"
            ),
            height=480,
        )

        st.markdown(
            "#### Walk-forward performans tablosu"
        )
        st.dataframe(
            wf_metrics,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### BIST100'e göre aktif performans"
        )
        st.dataframe(
            wf_active,
            use_container_width=True,
            hide_index=True,
        )

        wf_yearly = to_frame(
            client.get(
                "/api/backtest/walk-forward/yearly"
            )
        )

        if not wf_yearly.empty:
            wf_yearly = wf_yearly.set_index(
                "Year"
            )
            yearly_columns = [
                column
                for column in (
                    "Baseline_Robot",
                    "ML_Challenger",
                    "BIST100_Gross",
                )
                if column in wf_yearly.columns
            ]

            if yearly_columns:
                st.markdown(
                    "#### Walk-forward yıllık getiriler (%)"
                )
                render_bar_chart(
                    wf_yearly[yearly_columns],
                    yearly_columns,
                    title=(
                        "Walk-forward yıllık getiriler (%)"
                    ),
                )

        with st.expander(
            "Aylık yeniden eğitim denetim kaydı"
        ):
            st.dataframe(
                to_frame(
                    client.get(
                        "/api/backtest/"
                        "walk-forward/training-log"
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:
        st.info(
            "Walk-forward karşılaştırma henüz oluşturulmamış. "
            "Önce 15_walk_forward_ml_comparison_robot.ipynb "
            "notebook'unu çalıştır."
        )
        st.caption(str(error))


with tab_signals:
    title_col, button_col = st.columns(
        [4, 1]
    )
    title_col.subheader(
        "Baseline Robot ve ML Challenger günlük sinyalleri"
    )

    refresh_clicked = button_col.button(
        "Sinyalleri yenile",
        type="primary",
        use_container_width=True,
    )

    if refresh_clicked:
        try:
            with st.spinner(
                "Yahoo Finance verileri indiriliyor ve "
                "sinyaller hesaplanıyor..."
            ):
                client.post(
                    "/api/signals/refresh",
                    timeout=900,
                )
            st.success("Günlük sinyaller yenilendi.")
        except Exception as error:
            show_error(error)

    try:
        latest = client.get(
            "/api/signals/latest"
        )
        metadata = latest.get("metadata", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Sinyal tarihi",
            latest.get("signal_date", "—"),
        )
        m2.metric(
            "Son hisse tarihi",
            metadata.get("latest_stock_date", "—"),
        )
        m3.metric(
            "Veri gecikmesi",
            (
                f"{metadata.get('data_lag_calendar_days', 0)} gün"
            ),
        )
        m4.metric(
            "ML eşiği",
            (
                f"{metadata.get('probability_threshold', 0):.6f}"
                if metadata.get(
                    "probability_threshold"
                ) is not None
                else "—"
            ),
        )

        if metadata and not metadata.get(
            "trade_ready",
            True,
        ):
            st.warning(
                "Model-ready sinyal tarihi son hisse tarihinden "
                "eski. Emir kaydı oluşturmadan önce tarihleri "
                "kontrol et."
            )

        st.markdown("#### Günlük özet")
        st.dataframe(
            to_frame(latest.get("summary")),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### ML skor teşhisi")
        st.dataframe(
            to_frame(
                latest.get("model_diagnostic")
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Alış karşılaştırması")
        st.dataframe(
            to_frame(
                latest.get("buy_comparison")
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Satış karşılaştırması")
        st.dataframe(
            to_frame(
                latest.get("sell_comparison")
            ),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as error:
        show_error(error)



with tab_portfolio:
    st.subheader(
        "Alış / satış kaydı ve kâr-zarar takibi"
    )

    try:
        latest_portfolio_signals = client.get(
            "/api/signals/latest"
        )
    except Exception as error:
        latest_portfolio_signals = {}
        st.warning(
            "Günlük sinyal planı yüklenemedi. Manuel "
            "alış/satış kayıtları kullanılabilir."
        )
        st.caption(str(error))

    tracking_tabs = st.tabs(
        [
            "Baseline Robot",
            "ML Challenger",
        ]
    )

    with tracking_tabs[0]:
        render_portfolio_workspace(
            "Baseline_Robot",
            latest_portfolio_signals,
        )

    with tracking_tabs[1]:
        render_portfolio_workspace(
            "ML_Challenger",
            latest_portfolio_signals,
        )
