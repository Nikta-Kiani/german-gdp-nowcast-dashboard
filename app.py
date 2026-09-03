"""German GDP nowcast dashboard.

Run with:
    streamlit run app.py

Overview, Part I (indicator selection) and Part II (nowcasting results).
Data resolution (real vs bundled demo sample) is in ``src/dashboard/config.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dashboard.bootstrap import sync_real_data  # noqa: E402

sync_real_data()  # no-op unless a private data source is set in st.secrets

from dashboard import config as C  # noqa: E402
from dashboard import data as D  # noqa: E402
from dashboard import theme as T  # noqa: E402
from dashboard.sections import overview, selection, nowcasting  # noqa: E402

st.set_page_config(
    page_title="Nowcasting German GDP — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

T.register_plotly_template()
T.inject_css()

PAGES = {
    "Overview": overview.render,
    "Part I · Indicator selection": selection.render,
    "Part II · Nowcasting results": nowcasting.render,
}

with st.sidebar:
    st.markdown(
        f"<div style='font-weight:700;font-size:1.05rem;color:{C.INK};"
        "line-height:1.3'>Nowcasting German GDP</div>"
        f"<div style='color:{C.SUBTLE};font-size:0.82rem;margin-bottom:1rem'>"
        "Master's thesis companion</div>",
        unsafe_allow_html=True,
    )
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed",
                      key="nav")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{C.SUBTLE};font-size:0.78rem;line-height:1.5'>"
        "Pseudo-real-time expanding window<br>1991m01 → 2025m12<br>"
        "First-release GDP · 2011Q1–2025Q4</div>",
        unsafe_allow_html=True,
    )
    if C.IS_DEMO_DATA:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.info(
            "**Demo mode** — figures use a synthetic sample, not the thesis "
            "results. The live demo linked in the README uses the real cut.",
            icon="🧪",
        )

D.refresh_cache_if_stale()
PAGES[choice]()
