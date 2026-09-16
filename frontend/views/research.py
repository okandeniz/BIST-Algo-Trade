"""Advanced comparisons kept away from the daily workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import APIClient
from frontend.formatters import format_percent
from frontend.ui_helpers import (
    format_date,
    render_line_chart,
    show_error,
    to_frame,
)
from src.product_config import STRATEGIES


COLUMN_LABELS = {
    "Portfolio": "Strateji",
    "Start_Value": "Başlangıç değeri",
    "End_Value": "Bitiş değeri",
    "Total_Return_%": "Toplam getiri (%)",
    "CAGR_%": "Yıllık getiri (%)",
    "Max_Drawdown_%": "Maks. düşüş (%)",
    "Profit_Factor": "Kâr faktörü",
    "Win_Rate_%": "Kazanma oranı (%)",
    "Sharpe": "Sharpe",
    "Calmar": "Calmar",
    "Trade_Count": "İşlem sayısı",
    "Exposure_%": "Piyasada kalma (%)",
    "Alpha_Annual_%": "Yıllık alfa (%)",
    "Tracking_Error_%": "Takip hatası (%)",
    "Information_Ratio": "Bilgi oranı",
    "Active_Total_Return_pp": "Aktif getiri (puan)",
    "Upside_Capture_%": "Yükseliş yakalama (%)",
    "Downside_Capture_%": "Düşüş yakalama (%)",
    "Block": "Blok",
    "Block_Start": "Blok başlangıcı",
    "Block_End": "Blok bitişi",
    "Training_Event_Count": "Eğitim olayı",
    "Score_Coverage_%": "Skor kapsaması (%)",
    "Rows_With_Imputation_%": "Eksik veri tamamlanan (%)",
}


def _display_table(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if columns is not None:
        result = result[
            [column for column in columns if column in result.columns]
        ]
    if "Portfolio" in result:
        result["Portfolio"] = result["Portfolio"].map(
            lambda key: STRATEGIES[key].label
            if key in STRATEGIES
            else str(key).replace("_", " ")
        )
    numeric_columns = result.select_dtypes(include="number").columns
    result[numeric_columns] = result[numeric_columns].round(2)
    return result.rename(columns=COLUMN_LABELS)


def _render_walk_forward_details(
    walk_forward: dict,
    training: pd.DataFrame,
) -> None:
    metadata = walk_forward.get("metadata", {})
    model_labels = {
        "LogisticRegression": "Lojistik regresyon",
    }
    method_labels = {
        "Monthly expanding walk-forward": "Aylık genişleyen walk-forward",
    }
    frequency_labels = {
        "MS": "Aylık",
    }
    target_labels = {
        "Big_Winner_Label_2R": "2R büyük kazanan etiketi",
    }
    filter_labels = {
        "Quantile_Q40": "Kantil filtresi (Q40)",
    }

    threshold = metadata.get("probability_threshold")
    threshold_label = (
        format_percent(float(threshold) * 100)
        if threshold is not None
        else "—"
    )
    period_label = (
        f"{format_date(metadata.get('start'))} – "
        f"{format_date(metadata.get('end'))}"
    )

    overview = st.columns(4)
    overview[0].metric(
        "Yöntem",
        method_labels.get(
            metadata.get("method"),
            metadata.get("method", "—"),
        ),
    )
    overview[1].metric(
        "Model",
        model_labels.get(
            metadata.get("model"),
            metadata.get("model", "—"),
        ),
    )
    overview[2].metric("Olasılık eşiği", threshold_label)
    overview[3].metric(
        "Özellik sayısı",
        metadata.get("feature_count", "—"),
    )

    details = pd.DataFrame(
        [
            {
                "Ayar": "Değerlendirme dönemi",
                "Değer": period_label,
            },
            {
                "Ayar": "Model seçim dönemi",
                "Değer": metadata.get("selection_window", "—"),
            },
            {
                "Ayar": "Yeniden eğitim",
                "Değer": frequency_labels.get(
                    metadata.get("retrain_frequency"),
                    metadata.get("retrain_frequency", "—"),
                ),
            },
            {
                "Ayar": "Embargo",
                "Değer": f"{metadata.get('embargo_days', '—')} gün",
            },
            {
                "Ayar": "Hedef",
                "Değer": target_labels.get(
                    metadata.get("target"),
                    metadata.get("target", "—"),
                ),
            },
            {
                "Ayar": "Filtre",
                "Değer": filter_labels.get(
                    metadata.get("filter"),
                    metadata.get("filter", "—"),
                ),
            },
        ]
    )
    st.dataframe(details, width="stretch", hide_index=True)

    if metadata.get("warning"):
        st.warning(
            "2025 sonrası dönem daha önce incelendi. Bu sonuç, veri "
            "sızıntısı kontrollü bir audit çalışmasıdır; tamamen "
            "dokunulmamış bir test dönemi değildir."
        )

    active = to_frame(walk_forward.get("active_metrics"))
    if not active.empty:
        st.markdown("#### BIST100'e göre aktif performans")
        st.dataframe(
            _display_table(
                active,
                (
                    "Portfolio",
                    "Alpha_Annual_%",
                    "Active_Total_Return_pp",
                    "Information_Ratio",
                    "Tracking_Error_%",
                    "Upside_Capture_%",
                    "Downside_Capture_%",
                ),
            ),
            width="stretch",
            hide_index=True,
        )

    if not training.empty:
        st.markdown("#### Aylık eğitim blokları")
        st.dataframe(
            _display_table(
                training,
                (
                    "Block",
                    "Block_Start",
                    "Block_End",
                    "Training_Event_Count",
                    "Score_Coverage_%",
                    "Rows_With_Imputation_%",
                ),
            ),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Teknik veri"):
        st.caption("Ham metadata")
        st.json(metadata)
        if not training.empty:
            st.caption("Tüm eğitim alanları")
            st.dataframe(training, width="stretch", hide_index=True)


def _render_plan_summary(metadata: dict) -> None:
    summary = st.columns(4)
    summary[0].metric("Sinyal tarihi", format_date(metadata.get("signal_date")))
    summary[1].metric(
        "Son piyasa verisi",
        format_date(metadata.get("latest_stock_date")),
    )
    summary[2].metric(
        "İşleme hazır",
        "Evet" if metadata.get("trade_ready") else "Hayır",
    )
    summary[3].metric(
        "İndirme hatası",
        metadata.get("download_error_count", 0),
    )

    model_details = pd.DataFrame(
        [
            {
                "Ayar": "Model",
                "Değer": metadata.get("model_name", "—"),
            },
            {
                "Ayar": "Hedef",
                "Değer": metadata.get("model_target", "—"),
            },
            {
                "Ayar": "Filtre",
                "Değer": metadata.get("filter_name", "—"),
            },
            {
                "Ayar": "Olasılık eşiği",
                "Değer": (
                    format_percent(
                        float(metadata["probability_threshold"]) * 100
                    )
                    if metadata.get("probability_threshold") is not None
                    else "—"
                ),
            },
        ]
    )
    st.dataframe(model_details, width="stretch", hide_index=True)

    if metadata and not metadata.get("trade_ready", True):
        st.warning(
            "Sinyal tarihi son piyasa tarihinden eski. Güncel emir için "
            "önce Bugün sayfasından veriyi yenileyin."
        )

    with st.expander("Teknik plan verisi"):
        st.json(metadata)


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
                st.dataframe(
                    _display_table(
                        metrics,
                        (
                            "Portfolio",
                            "End_Value",
                            "Total_Return_%",
                            "CAGR_%",
                            "Max_Drawdown_%",
                            "Sharpe",
                            "Calmar",
                            "Trade_Count",
                            "Exposure_%",
                        ),
                    ),
                    width="stretch",
                    hide_index=True,
                )

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
                training = to_frame(
                    client.get("/api/backtest/walk-forward/training-log")
                )
                _render_walk_forward_details(walk_forward, training)
        except Exception as error:
            show_error(error)
            st.caption("Araştırma artifact'leri üretilmemiş veya erişilemiyor olabilir.")

    with diagnostics_tab:
        try:
            latest = client.get("/api/signals/latest")
            metadata = latest.get("metadata", {})
            st.markdown("#### Son plan özeti")
            _render_plan_summary(metadata)
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
