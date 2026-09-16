"""Streamlit entry point for the simplified BIST100 Robot dashboard."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import APIClient
from frontend.setup_guide import render_connection_help, render_setup_guide
from frontend.ui_helpers import format_date
from frontend.views.performance import render_performance
from frontend.views.portfolio import render_portfolio
from frontend.views.research import render_research
from frontend.views.today import render_today


st.set_page_config(
    page_title="BIST100 Robot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clear_cached_data() -> None:
    """Single hook used by mutating views; ready for cached reads later."""
    st.cache_data.clear()


default_api_url = os.getenv("API_URL", "http://127.0.0.1:8000")

st.sidebar.title("BIST100 Robot")
st.sidebar.caption("Günlük paper-trading çalışma alanı")

with st.sidebar.expander("Bağlantı ayarları"):
    api_url = st.text_input("FastAPI adresi", value=default_api_url)

client = APIClient(api_url)

try:
    health = client.get("/health", timeout=5)
except Exception as error:
    st.sidebar.error("Backend bağlantısı yok")
    render_connection_help(error)
    st.stop()

setup = health.get("setup", {"ready": True, "checks": []})
setup_ready = setup.get("ready", True)
pages = ["Bugün", "Portföy", "Performans", "Araştırma"]
if not setup_ready:
    pages.insert(0, "Başlangıç")

page = st.sidebar.radio(
    "Navigasyon",
    pages,
    label_visibility="collapsed",
)

if setup_ready:
    st.sidebar.success("Sistem hazır")
else:
    st.sidebar.warning(
        f"{setup.get('blocking_count', 0)} kurulum adımı bekliyor"
    )

latest_market_date = health.get("market_data", {}).get("latest_date")
st.sidebar.caption(f"Son piyasa verisi: {format_date(latest_market_date)}")
release = health.get("release", {})
st.sidebar.caption(f"Release: {release.get('release_id') or '—'}")

if page == "Başlangıç":
    render_setup_guide(health, client, clear_cached_data)
elif page == "Bugün":
    render_today(client, clear_cached_data)
elif page == "Portföy":
    render_portfolio(
        client,
        clear_cached_data,
        health.get("trading_config", {}),
    )
elif page == "Performans":
    render_performance(client)
else:
    render_research(client)

st.sidebar.divider()
st.sidebar.caption("Eğitim, araştırma ve paper-trading amaçlıdır.")
