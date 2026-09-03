"""Part I — Indicator selection views.

Every method exposes a different native signal (binary inclusion for the linear
screens, mean |SHAP| for XGBoost, fixed membership for ifoCAST). They are
harmonised onto a common per-series importance mass and read two ways: raw
category mass share, and the deviation from availability (selected share minus
the category's share of the candidate universe). The second view matters
because the universe is survey-heavy, so raw shares partly echo what exists
rather than what a method prefers.
"""

from __future__ import annotations

import streamlit as st

from .. import charts
from .. import config as C
from .. import data as D
from .. import theme as T


def render() -> None:
    T.eyebrow("Part I")
    st.markdown("# Indicator selection")
    st.markdown(
        "Four statistical signals are compared with a fixed 19-series ifoCAST "
        "reference. The methods estimate different objects, so the common "
        "comparison is their share of importance mass by economic category."
    )
    tab1, tab2, tab3, tab4 = st.tabs([
        "Composition over time",
        "Regime averages",
        "Agreement",
        "Timing & ragged edge",
    ])

    with tab1:
        _structural_shift()
    with tab2:
        _regime_switching()
    with tab3:
        _consensus()
    with tab4:
        _ragged_edge()


def _structural_shift() -> None:
    st.markdown("### Time-varying category emphasis")
    methods = [m for m in D.available_methods()
               if C.SELECTION_METHODS[m]["kind"] != "fixed"]
    default = "EN (raw)" if "EN (raw)" in methods else 0
    method = st.selectbox(
        "Method", methods,
        index=methods.index(default) if isinstance(default, str) else default,
        format_func=C.selection_label,
        help="Elastic net is the primary screen (COVID-weighted, capped at 60 "
             "indicators per origin). XGBoost is a non-linear cross-check.",
    )

    share = D.category_share_over_time_any(method)
    if share.empty:
        st.warning("Time-varying selection is unavailable for this method.")
        return

    soft_hard = D.soft_hard_share_over_time(method)
    st.plotly_chart(
        charts.soft_hard_lines(
            soft_hard, D.universe_soft_hard(),
            title=f"{C.selection_label(method)} — soft vs hard data over time"),
        width="stretch")
    T.callout(
        "<b>Thesis finding.</b> Every statistical method puts most of its mass "
        "on delayed hard activity: 65–100% across methods and regimes, versus "
        "29.1% of the candidate panel. This is completed-quarter predictive "
        "association, not a claim that surveys are unhelpful early in a quarter."
    )

    with st.expander("Why the universe base rate matters"):
        st.plotly_chart(
            charts.universe_bar(D.universe_category_share()),
            width="stretch",
        )
        st.caption(
            "Surveys are 66.8% of the 585-series universe. Selection shares "
            "must therefore be read relative to what was available to select."
        )

    with st.expander("Full 11-category composition (stacked area)"):
        st.plotly_chart(charts.structural_shift_area(share), width="stretch")
        st.caption(
            "Raw composition of the selected set (bands sum to 100%). The wide "
            "survey band partly reflects the survey-heavy universe; use the "
            "base-rate lines above before interpreting category emphasis."
        )

    with st.expander("Non-linear cross-check: XGBoost"):
        for m, note in [("XGBoost (SHAP)", "mean |SHAP| mass")]:
            if m == method:
                continue
            sh = D.soft_hard_share_over_time(m)
            if sh.empty:
                st.info(f"{C.selection_label(m)} importances unavailable.")
                continue
            st.plotly_chart(
                charts.soft_hard_lines(
                    sh, D.universe_soft_hard(),
                    title=f"{C.selection_label(m)} — soft vs hard emphasis ({note})",
                    height=340),
                width="stretch")


def _regime_switching() -> None:
    st.markdown("### Regime-level category emphasis")
    T.callout(
        "Regime averages summarise composition; they do not establish a break. "
        "The hard-data concentration is stable. The apparent COVID rotation "
        "toward surveys is brief within the elastic net and is not reproduced "
        "by the other methods."
    )
    long = D.regime_soft_hard()
    if long.empty:
        st.warning("Regime data unavailable.")
    else:
        st.plotly_chart(
            charts.regime_rotation_bars(long, D.universe_soft_hard()),
            width="stretch",
        )
        T.callout(
            "Compare levels with the dotted universe shares, and movements "
            "across panels, separately. The shared result is the hard-data "
            "level; the regime-to-regime movement is method-dependent."
        )

    st.markdown("#### Full category composition by regime")
    full = D.regime_category_share()
    if full.empty:
        st.info("Regime composition unavailable.")
    else:
        methods = full["method"].unique().tolist()
        st.plotly_chart(charts.regime_share_bars(full, methods), width="stretch")
        st.caption(
            "Shares of selected mass before availability adjustment. Read "
            "shifts across panels (regimes) rather than the absolute level of "
            "the survey bar, which reflects the survey-heavy universe."
        )


def _relabel_agreement(df):
    """Replace internal method keys with thesis-facing labels."""
    mapper = {k: C.selection_label(k) for k in df.index}
    return df.rename(index=mapper, columns=mapper)


def _consensus() -> None:
    st.markdown("### Method agreement diagnostics")
    summary = D.en_stability_summary()
    if summary:
        T.stat_cards([
            (f"{summary['persistent_series']}",
             "elastic-net series selected at every origin"),
            (f"{summary['mean_ifocast_jaccard']:.3f}",
             "mean EN–ifoCAST Jaccard"),
            (f"{summary['n_ifocast']}", "active ifoCAST predictors"),
        ])

    df = D.cross_method_agreement()
    st.plotly_chart(
        charts.agreement_heatmap(
            _relabel_agreement(df),
            "Cross-method agreement — Spearman rank correlation",
            "Spearman ρ"),
        width="stretch")
    T.callout(
        "<b>Thesis finding.</b> Data-driven rank correlations remain in "
        "0.28–0.46, and correlations with the fixed ifoCAST set are lower "
        "still (0.14–0.23). Methods agree on the hard-activity block, not on "
        "a universal series list."
    )

    with st.expander("Exploratory top-20 overlap"):
        jaccard = D.method_overlap_jaccard(top_n=20)
        st.plotly_chart(
            charts.agreement_heatmap(
                _relabel_agreement(jaccard),
                "Exploratory Jaccard overlap of aggregate top-20 indicators",
                "Jaccard"),
            width="stretch")
        st.caption(
            "This dashboard-only diagnostic compares aggregate top-20 lists. "
            "It is not the thesis mean per-origin EN–ifoCAST Jaccard statistic."
        )


def _ragged_edge() -> None:
    st.markdown("### Timing and predictive content answer different questions")
    df = D.load_ragged_edge()
    if df.empty:
        st.warning("Ragged-edge diagnostics unavailable.")
        return

    n_total = len(df)
    n_timely = int((df["pub_lag"] == 0).sum())
    n_delayed = int((df["pub_lag"] >= 1).sum())
    n_filled = int((df["status"] == "ar_filled").sum())
    T.stat_cards([
        (f"{n_total}", "series at origin"),
        (f"{n_timely}", "available same month (lag 0)"),
        (f"{n_delayed}", "delayed (lag ≥ 1)"),
        (f"{n_filled}", "AR-backfilled at the edge"),
    ])

    mat = D.publag_category_matrix()
    if not mat.empty:
        st.plotly_chart(charts.publag_composition(mat), width="stretch")

    T.callout(
        "Part I scores association with <b>completed-quarter</b> growth and does "
        "not reward early publication; XGBoost–SHAP is the exception because it "
        "is recorded inside the real-time nowcast loop. Part II masks unreleased "
        "values and fills the edge using univariate AR models estimated only on "
        "released history. Timely surveys and delayed hard data are therefore "
        "complements, not substitutes."
    )
