"""Landing page: thesis framing, the two-part workflow and headline numbers."""

from __future__ import annotations

import streamlit as st

from .. import config as C
from .. import theme as T


def render() -> None:
    T.hero(
        "Nowcasting & Indicator Selection in a Data-Rich Environment",
        "An application to German GDP growth — real-time indicator screening "
        "and short-horizon nowcasting over 2011–2025.",
        eyebrow="M.Sc. Thesis · Statistics & Econometrics",
    )

    T.stat_cards([
        ("585", "candidate indicators"),
        ("11", "economic categories"),
        ("1991–2025", "monthly sample"),
        ("180", "real-time origins"),
        ("9", "nowcasting models"),
    ])

    col1, col2 = st.columns(2, gap="large")
    with col1:
        T.eyebrow("Part I")
        st.markdown("### Indicator selection")
        st.markdown(
            "A real-time, expanding-window screen of a high-dimensional German "
            "macro panel. Indicators are re-selected at every origin from "
            "**2011m01** onward, on predictors aggregated to quarterly "
            "frequency.\n\n"
            "- **Elastic Net** — primary data-driven screen\n"
            "- **Block-balanced (k=20)** — EN with structural breadth (≥1 per category)\n"
            "- **PLS** — supervised dimensionality reduction (Part I comparison only)\n"
            "- Cross-checked against **XGBoost (SHAP)** importances"
        )
    with col2:
        T.eyebrow("Part II")
        st.markdown("### Nowcasting")
        st.markdown(
            "The selected indicators feed a focused suite of nowcasting models, "
            "evaluated out-of-sample at the final (M3) information set across "
            "three economic regimes. The headline DFM models keep the monthly "
            "panel in mixed frequency (Mariano–Murasawa encoding) after "
            "publication-lag masking and AR ragged-edge fill; XGBoost "
            "additionally aggregates indicators to quarterly frequency before "
            "fitting, and MLP-Factor reuses the DFM's mixed-frequency factors.\n\n"
            "- **DFM (A-CD-TPN)** on EN, ifoCAST fixed set, and block-balanced inputs\n"
            "- **DFM-SV (integrated, k=2)** — same EN inputs, with stochastic "
            "volatility fed back into the Kalman smoother, so the point nowcast "
            "can differ slightly from DFM-EN while intervals stay calibrated\n"
            "- **Equal-weight combination** of the main DFM variants\n"
            "- **XGBoost** and a **factor-augmented MLP** as ML benchmarks\n"
            "- Classical **AR(1)** and **Random-Walk** baselines"
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    T.eyebrow("How to read this dashboard")
    st.markdown("### Three regimes anchor every comparison")
    rc = st.columns(3, gap="large")
    regimes = [
        ("pre-COVID", "2011Q1 – 2019Q4",
         "The pre-pandemic evaluation window (36 quarters)."),
        ("COVID", "2020Q1 – 2021Q4",
         "The pandemic shock window (8 quarters); squared-error metrics are "
         "dominated by a few extreme misses."),
        ("post-COVID", "2022Q1 – 2025Q4",
         "The post-pandemic window (16 quarters), spanning the inflation and "
         "normalisation phase."),
    ]
    for c, (name, span, desc) in zip(rc, regimes):
        with c:
            st.markdown(
                f"<div style='border-top:4px solid {C.REGIME_COLORS[name]};"
                f"padding-top:0.6rem'><b>{name}</b><br>"
                f"<span style='color:{C.SUBTLE};font-size:0.85rem'>{span}</span>"
                f"<p style='font-size:0.9rem;margin-top:0.4rem'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr/>", unsafe_allow_html=True)
    T.eyebrow("Methodology at a glance")
    st.markdown("### The end-to-end pipeline")
    st.markdown(
        "The thesis follows two linked pipelines. Part I screens the 585-series "
        "panel at 180 expanding-window origins and outputs time-varying "
        "indicator sets; Part II turns those sets into real-time GDP nowcasts "
        "on the same evaluation grid. Part I aggregates predictors to quarterly "
        "frequency via a raw-level bridge (back-transform to raw monthly levels, "
        "quarterly mean, re-transform to growth rates). Part II keeps the monthly "
        "panel and embeds it in a mixed-frequency state space (Mariano–Murasawa "
        "encoding), with publication-lag masking and AR ragged-edge fill "
        "enforcing the information set available at each origin. Expand either "
        "diagram below for the full sequence."
    )
    fc1, fc2 = st.columns(2, gap="large")
    with fc1:
        st.markdown(
            f"<span class='flow-tag'>Part I</span> "
            f"<b style='color:{C.INK}'>Indicator selection</b>",
            unsafe_allow_html=True)
        T.selection_flow()
    with fc2:
        st.markdown(
            f"<span class='flow-tag'>Part II</span> "
            f"<b style='color:{C.INK}'>Nowcasting</b>",
            unsafe_allow_html=True)
        T.nowcast_flow()

    T.callout(
        "Use the sidebar to move through the workflow in order: "
        "<b>Part I — Indicator selection</b> shows <i>what</i> the data tells us "
        "to watch; <b>Part II — Nowcasting</b> shows <i>how well</i> those "
        "indicators translate into GDP nowcasts."
    )
