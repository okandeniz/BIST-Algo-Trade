"""The daily operator view."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from frontend.api_client import APIClient
from src.product_config import PRIMARY_STRATEGY, STRATEGIES
from frontend.ui_helpers import format_date, show_error, to_frame


def render_today(
    client: APIClient,
    clear_cache: Callable[[], Any],
) -> None:
    strategy = STRATEGIES[PRIMARY_STRATEGY]
    st.title("Bugün")
    st.caption(
        "Günlük veri durumunu kontrol edin ve Baseline Robot aksiyonlarını tek ekrandan yönetin."
    )

    try:
        latest = client.get("/api/signals/latest")
    except Exception as error:
        latest = {}
        st.warning("Henüz kullanılabilir bir günlük plan bulunmuyor.")
        st.caption(str(error))

    metadata = latest.get("metadata", {})
    title_column, action_column = st.columns([4, 1.3])
    title_column.subheader("Günlük durum")

    if action_column.button(
        "Günü güncelle",
        type="primary",
        width="stretch",
        help="Piyasa verisini indirir ve üç stratejinin planlarını yeniden üretir.",
    ):
        try:
            with st.spinner("Veriler ve sinyaller güncelleniyor…"):
                client.post("/api/signals/refresh", timeout=900)
            clear_cache()
            st.success("Günlük plan güncellendi.")
            st.rerun()
        except Exception as error:
            show_error(error)

    status_columns = st.columns(4)
    status_columns[0].metric("Sinyal tarihi", format_date(latest.get("signal_date")))
    status_columns[1].metric(
        "Son piyasa verisi",
        format_date(metadata.get("latest_stock_date")),
    )
    status_columns[2].metric(
        "Veri gecikmesi",
        f"{int(metadata.get('data_lag_calendar_days', 0))} gün",
    )
    status_columns[3].metric("Aktif strateji", strategy.label)

    if metadata and not metadata.get("trade_ready", True):
        st.warning(
            "Sinyal tarihi son piyasa tarihinden eski. Bu planı güncel emir olarak kullanmadan önce veriyi yenileyin."
        )

    buys = to_frame(latest.get(strategy.buy_plan_key))
    sells = to_frame(latest.get(strategy.sell_plan_key))

    st.subheader("Bugünün aksiyonları")
    buy_count = len(buys)
    sell_count = len(sells)
    a1, a2, a3 = st.columns(3)
    a1.metric("Alış adayı", buy_count)
    a2.metric("Çıkış adayı", sell_count)
    a3.metric("Toplam aksiyon", buy_count + sell_count)

    buy_column, sell_column = st.columns(2)
    with buy_column:
        st.markdown("#### Alış adayları")
        if buys.empty:
            st.success("Bugün yeni alış adayı yok.")
        else:
            preferred = [
                "Ticker",
                "Score",
                "Estimated_Shares",
                "Estimated_Entry",
                "Estimated_Stop",
                "Reason",
            ]
            st.dataframe(
                buys[[column for column in preferred if column in buys]],
                width="stretch",
                hide_index=True,
            )

    with sell_column:
        st.markdown("#### Çıkış adayları")
        if sells.empty:
            st.success("Bugün yeni çıkış adayı yok.")
        else:
            preferred = ["Ticker", "Reason", "Exit_Price", "Stop_Loss"]
            st.dataframe(
                sells[[column for column in preferred if column in sells]],
                width="stretch",
                hide_index=True,
            )

    st.info(
        "İşlem kaydı oluşturmak için Portföy sayfasını kullanın. RS126 ve ML sonuçları Araştırma sayfasında karşılaştırma amacıyla tutulur."
    )

    with st.expander("Plan özeti"):
        summary = to_frame(latest.get("summary"))
        if summary.empty:
            st.caption("Özet bulunmuyor.")
        else:
            st.dataframe(summary, width="stretch", hide_index=True)
