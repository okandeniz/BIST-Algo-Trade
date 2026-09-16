"""Guided first-run and degraded-system screens."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from frontend.api_client import APIClient


STATUS_ICON = {
    "ready": "✅",
    "warning": "🟡",
    "error": "❌",
    "info": "ℹ️",
}


def render_connection_help(error: Exception) -> None:
    st.title("Başlangıç kontrolü")
    st.error("Uygulama servisine bağlanılamıyor.")
    st.markdown(
        "Açık uygulamayı kapatın ve proje klasöründe "
        "tek komutla yeniden başlatın:"
    )
    st.code(".\\run_app.ps1", language="powershell")
    st.caption(
        "Bu komut backend'i ve arayüzü doğru sırayla başlatır."
    )
    with st.expander("Teknik ayrıntı"):
        st.code(str(error), language=None)
        st.markdown(
            "Arayüzü geliştirici modunda ayrı çalıştırıyorsanız "
            "backend'i `.\\run_backend.ps1` ile başlatın."
        )


def render_setup_guide(
    health: dict[str, Any],
    client: APIClient,
    clear_cache: Callable[[], Any],
) -> None:
    setup = health.get("setup", {})
    checks = setup.get("checks", [])

    st.title("Başlangıç kontrolü")
    if setup.get("ready", False):
        st.success("Günlük kullanım için zorunlu bileşenler hazır.")
    else:
        st.warning(
            "Devam etmeden önce aşağıdaki kırmızı maddeleri "
            "tamamlayın. Diğer sayfaları yine inceleyebilirsiniz."
        )

    status_by_key = {
        check.get("key"): check.get("status")
        for check in checks
    }

    for check in checks:
        status = check.get("status", "info")
        icon = STATUS_ICON.get(status, "ℹ️")
        with st.container(border=True):
            st.markdown(f"### {icon} {check.get('label', 'Kontrol')}")
            st.write(check.get("detail", ""))
            if status == "error" and check.get("path"):
                st.caption("Beklenen konum")
                st.code(str(check["path"]), language=None)

            if check.get("key") == "release" and status == "error":
                missing = setup.get("missing_artifacts", [])
                if missing:
                    with st.expander("Eksik artifact anahtarları"):
                        st.code("\n".join(missing), language=None)
                st.caption(
                    "Araştırma çıktılarını yeniden üretmek için "
                    "notebooks/active akışını sırasıyla çalıştırın."
                )

    can_refresh = all(
        status_by_key.get(key) == "ready"
        for key in ("database", "ticker_file", "release")
    )
    needs_refresh = any(
        status_by_key.get(key) != "ready"
        for key in ("market_data", "daily_plan")
    )

    if can_refresh and needs_refresh:
        st.subheader("Otomatik hazırlık")
        st.caption(
            "Piyasa verisini indirip bugünün işlem planını oluşturur. "
            "İşlem internet bağlantısına göre birkaç dakika sürebilir."
        )
        if st.button(
            "Veriyi ve sinyalleri hazırla",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("Veriler indiriliyor ve plan hazırlanıyor..."):
                    client.post("/api/signals/refresh", timeout=900)
                clear_cache()
                st.success("Hazırlık tamamlandı.")
                st.rerun()
            except Exception as error:
                st.error("Otomatik hazırlık tamamlanamadı.")
                with st.expander("Teknik ayrıntı"):
                    st.code(str(error), language=None)
