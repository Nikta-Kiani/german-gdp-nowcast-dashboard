"""Landing page: thesis framing, the two-part workflow and headline numbers."""

from __future__ import annotations

import streamlit as st

from .. import config as C
from .. import theme as T


def render() -> None:
    T.hero(
        "Nowcasting & Indicator Selection in a Data-Rich Environment",
        "Indicator selection and pseudo-real-time nowcasting, evaluated on "
        "first-release GDP growth from 2011Q1 to 2025Q4.",
        eyebrow="M.Sc. Thesis · Statistics & Data Science",
    )

    T.stat_cards([
        ("585", "candidate indicators"),
        ("180", "monthly origins"),
        ("60", "evaluation quarters"),
        ("11", "headline candidates"),
        ("3", "calendar-defined regimes"),
    ])

    T.eyebrow("What the thesis finds")
    st.markdown("### Three results organise the dashboard")
    findings = st.columns(3, gap="large")
    with findings[0]:
        st.markdown("**Selection agrees on categories, not series**")
        st.markdown(
            "All four statistical signals favour delayed hard-activity data, "
            "but their series-level rank correlations remain below **0.5**."
        )
    with findings[1]:
        st.markdown("**Performance changes with the regime**")
        st.markdown(
            "Monthly factor models contain the COVID swings better than quarterly "
            "AR benchmarks. After 2022, short-memory and adaptive forecasts lead "
            "in point estimates."
        )
    with findings[2]:
        st.markdown("**The sample does not identify a champion**")
        st.markdown(
            "The equal-weight DFM combination has the lowest full-sample RMSFE, "
            "but the 90% model confidence set retains all eleven candidates."
        )

    T.callout(
        "<b>Scope.</b> This is a pseudo-real-time exercise: publication lags and "
        "first-release GDP are respected, but historical predictor revisions are "
        "not reconstructed. Rankings are point estimates unless a statistical "
        "test is stated."
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
         "The low-growth post-2022 window (16 quarters); estimates remain "
         "sensitive to a few observations."),
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
    st.markdown("### One protocol, two linked questions")
    st.markdown(
        "Part I asks which indicators are associated with **completed-quarter** "
        "growth. Part II asks how those inputs perform when releases arrive in "
        "real time. Selection is refreshed when a completed quarter enters the "
        "training sample; within a target quarter, the indicator list stays fixed "
        "while the available information grows from M1 to M3."
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
        "Use <b>Part I</b> for selection composition and agreement. Use "
        "<b>Part II</b> for accuracy, information accrual, formal tests, "
        "uncertainty and fitted-model interpretation."
    )
